package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/theapemachine/errnie"
)

/*
StoreMemory stores a memory with embedding.
It creates a new memory entry with the provided content and metadata,
stores it in the appropriate memory stores, and returns the unique ID.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - agentID: The ID of the agent who owns this memory
  - content: The textual content of the memory
  - memType: The type of memory (personal or global)
  - source: The source of the memory (conversation, document, etc.)
  - metadata: Additional information about the memory

Returns:
  - The unique ID of the stored memory
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) StoreMemory(ctx context.Context, agentID string, content string, memType MemoryType, source string, metadata map[string]interface{}) (string, error) {
	// Generate a unique ID for the memory
	memoryID := uuid.New().String()

	// Set defaults for metadata
	if metadata == nil {
		metadata = make(map[string]interface{})
	}

	// Get embedding from the provider
	var embedding []float32
	var err error

	if um.embeddingProvider != nil {
		embedding, err = um.embeddingProvider.GetEmbedding(ctx, content)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to get embedding for memory: %v", err))
		}
	}

	// Create the memory entry
	entry := &EnhancedMemoryEntry{
		ID:          memoryID,
		AgentID:     agentID,
		Content:     content,
		Embedding:   embedding,
		Type:        memType,
		Source:      source,
		CreatedAt:   time.Now(),
		AccessCount: 0,
		LastAccess:  time.Now(),
		Metadata:    metadata,
	}

	// Store in the in-memory map
	um.mutex.Lock()
	um.memoryData[memoryID] = entry
	um.mutex.Unlock()

	// Store in vector store if available
	if um.options.EnableVectorStore && um.vectorStore != nil && len(embedding) > 0 {
		// Prepare payload for vector store
		payload := map[string]interface{}{
			"agent_id":   agentID,
			"content":    content,
			"type":       string(memType),
			"source":     source,
			"created_at": entry.CreatedAt,
			"metadata":   metadata,
		}

		err := um.vectorStore.StoreVector(ctx, memoryID, embedding, payload)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to store memory in vector store: %v", err))
		}
	}

	// Store in graph store if available
	if um.options.EnableGraphStore && um.graphStore != nil {
		// Create node properties
		properties := map[string]interface{}{
			"agent_id":   agentID,
			"content":    content,
			"type":       string(memType),
			"source":     source,
			"created_at": entry.CreatedAt.Format(time.RFC3339),
		}

		// Add metadata to properties
		for k, v := range metadata {
			properties[k] = v
		}

		// Create labels based on memory type
		labels := []string{"Memory"}
		if memType == MemoryTypePersonal {
			labels = append(labels, "Personal")
		} else {
			labels = append(labels, "Global")
		}

		err := um.graphStore.CreateNode(ctx, memoryID, labels, properties)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to store memory in graph store: %v", err))
		}
	}

	// For simplicity, also store in the base memory store
	err = um.baseStore.Store(ctx, "memory:"+memoryID, content)
	if err != nil {
		errnie.Info(fmt.Sprintf("Failed to store memory in base store: %v", err))
	}

	return memoryID, nil
}

/*
ExtractMemories extracts important information from text that should be remembered.
It analyzes text to identify significant information worth storing as memories.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - agentID: The ID of the agent who will own these memories
  - text: The text to extract memories from
  - source: The source of the text

Returns:
  - A slice of strings containing the extracted memories
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) ExtractMemories(ctx context.Context, agentID string, text string, source string) ([]string, error) {
	// If the text is too short, don't extract any memories
	if len(text) < 50 {
		return []string{}, nil
	}

	// Use OpenAI API to extract memories
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	type OpenAIRequest struct {
		Model     string    `json:"model"`
		Messages  []Message `json:"messages"`
		MaxTokens int       `json:"max_tokens"`
	}

	type OpenAIResponse struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	// Create a system prompt for memory extraction
	systemPrompt := `You are a memory extraction system for an AI agent. Your job is to identify important information in the provided text that should be stored as memories.
Extract 2-5 concise, important facts or insights that would be valuable to remember later. Focus on factual information, key insights, and important details.
Format each memory as a complete, standalone statement on a new line.
Do not include numbering, bullets, or any other formatting.
Do not include any explanations or commentary.`

	userPrompt := fmt.Sprintf("Text to extract memories from: %s", text)

	// Create the request
	reqBody := OpenAIRequest{
		Model: "gpt-3.5-turbo",
		Messages: []Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		MaxTokens: 500,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		errnie.Error(fmt.Errorf("failed to marshal request for memory extraction: %w", err))
		// Fall back to basic extraction if LLM call fails
		return um.basicExtractMemories(ctx, agentID, text, source)
	}

	// Create HTTP request to OpenAI API
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		errnie.Error(fmt.Errorf("failed to create HTTP request for memory extraction: %w", err))
		return um.basicExtractMemories(ctx, agentID, text, source)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_KEY"))

	// Execute the request
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		errnie.Error(fmt.Errorf("failed to execute HTTP request for memory extraction: %w", err))
		return um.basicExtractMemories(ctx, agentID, text, source)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		errnie.Error(fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(bodyBytes)))
		return um.basicExtractMemories(ctx, agentID, text, source)
	}

	// Parse the response
	var openaiResp OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		errnie.Error(fmt.Errorf("failed to decode OpenAI response: %w", err))
		return um.basicExtractMemories(ctx, agentID, text, source)
	}

	if len(openaiResp.Choices) == 0 {
		errnie.Error(fmt.Errorf("no choices in OpenAI response"))
		return um.basicExtractMemories(ctx, agentID, text, source)
	}

	// Extract the memories from the response
	extractedContent := openaiResp.Choices[0].Message.Content
	memories := strings.Split(strings.TrimSpace(extractedContent), "\n")

	// Filter out empty memories and store them
	var validMemories []string
	for _, memory := range memories {
		memory = strings.TrimSpace(memory)
		if memory == "" {
			continue
		}

		// Store the memory
		memoryID, err := um.StoreMemory(ctx, agentID, memory, MemoryTypePersonal, source, nil)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to store extracted memory: %v", err))
			continue
		}
		validMemories = append(validMemories, memoryID)
	}

	return validMemories, nil
}

/*
basicExtractMemories provides a fallback extraction method when LLM is unavailable.
It uses simple heuristics to extract potential memories from text.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - agentID: The ID of the agent who will own these memories
  - text: The text to extract memories from
  - source: The source of the text

Returns:
  - A slice of strings containing the extracted memories
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) basicExtractMemories(ctx context.Context, agentID string, text string, source string) ([]string, error) {
	// Split text into paragraphs
	paragraphs := strings.Split(text, "\n\n")
	var memories []string
	var memoryIDs []string

	for _, paragraph := range paragraphs {
		paragraph = strings.TrimSpace(paragraph)
		if len(paragraph) < 20 {
			continue // Skip short paragraphs
		}

		// Store paragraphs that might contain valuable information
		// Look for sentences that might be interesting facts or insights
		sentences := strings.Split(paragraph, ".")
		for _, sentence := range sentences {
			sentence = strings.TrimSpace(sentence)
			if len(sentence) < 20 || len(sentence) > 300 {
				continue // Skip very short or very long sentences
			}

			// Look for indicators of important information
			lowerSentence := strings.ToLower(sentence)
			if strings.Contains(lowerSentence, "important") ||
				strings.Contains(lowerSentence, "key") ||
				strings.Contains(lowerSentence, "significant") ||
				strings.Contains(lowerSentence, "critical") ||
				strings.Contains(lowerSentence, "essential") ||
				strings.Contains(lowerSentence, "note that") {
				memories = append(memories, sentence)
			}
		}

		// If the paragraph is of reasonable length, consider it a potential memory
		if len(memories) == 0 && len(paragraph) > 50 && len(paragraph) < 300 {
			memories = append(memories, paragraph)
		}
	}

	// Limit the number of memories to extract
	maxMemories := 5
	if len(memories) > maxMemories {
		memories = memories[:maxMemories]
	}

	// Store the extracted memories
	for _, memory := range memories {
		memoryID, err := um.StoreMemory(ctx, agentID, memory, MemoryTypePersonal, source, nil)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to store extracted memory: %v", err))
			continue
		}
		memoryIDs = append(memoryIDs, memoryID)
	}

	return memoryIDs, nil
}

/*
GenerateMemoryQueries generates semantic search queries from a user prompt.
It uses an LLM to analyze the user query and generate targeted search queries
that will help retrieve the most relevant memories.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - query: The original user query or prompt

Returns:
  - A slice of search queries to retrieve relevant memories
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) GenerateMemoryQueries(ctx context.Context, query string) ([]string, error) {
	// Avoid API calls for very short queries
	if len(query) < 10 {
		return []string{query}, nil
	}

	// Use OpenAI API to generate memory queries
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	type OpenAIRequest struct {
		Model     string    `json:"model"`
		Messages  []Message `json:"messages"`
		MaxTokens int       `json:"max_tokens"`
	}

	type OpenAIResponse struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	// Create a system prompt for query generation
	systemPrompt := `You are a memory query generator for an AI assistant. Your job is to analyze the user's query and generate 3-5 search queries that will help retrieve relevant information from the assistant's memory.
The queries should be diverse to cover different aspects of what might be relevant.
Each query should be a short phrase or question that captures an important aspect of the information needed.
Format each query on a separate line with no numbering, bullets, or other formatting.`

	userPrompt := fmt.Sprintf("User query: %s", query)

	// Create the request
	reqBody := OpenAIRequest{
		Model: "gpt-4o-mini",
		Messages: []Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		MaxTokens: 300,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		errnie.Error(fmt.Errorf("failed to marshal request for query generation: %w", err))
		// Fall back to using the original query
		return []string{query}, nil
	}

	// Create HTTP request to OpenAI API
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		errnie.Error(fmt.Errorf("failed to create HTTP request for query generation: %w", err))
		return []string{query}, nil
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_KEY"))

	// Execute the request
	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		errnie.Error(fmt.Errorf("failed to execute HTTP request for query generation: %w", err))
		return []string{query}, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		errnie.Error(fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(bodyBytes)))
		return []string{query}, nil
	}

	// Parse the response
	var openaiResp OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		errnie.Error(fmt.Errorf("failed to decode OpenAI response: %w", err))
		return []string{query}, nil
	}

	if len(openaiResp.Choices) == 0 {
		errnie.Error(fmt.Errorf("no choices in OpenAI response"))
		return []string{query}, nil
	}

	// Extract the queries from the response
	extractedContent := openaiResp.Choices[0].Message.Content
	queries := strings.Split(strings.TrimSpace(extractedContent), "\n")

	// Filter out empty queries
	var validQueries []string
	for _, q := range queries {
		q = strings.TrimSpace(q)
		if q != "" {
			validQueries = append(validQueries, q)
		}
	}

	// Always include the original query
	validQueries = append(validQueries, query)

	return validQueries, nil
}

/*
PrepareContext enriches a prompt with relevant memories.
It retrieves memories relevant to the query and combines them with the
original query using the configured template.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - agentID: The ID of the agent to retrieve memories for
  - query: The original query text

Returns:
  - The enriched context with relevant memories
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) PrepareContext(ctx context.Context, agentID string, query string) (string, error) {
	// Default to just returning the query if no vector store
	if !um.options.EnableVectorStore || um.vectorStore == nil {
		return query, nil
	}

	// Generate memory queries using LLM
	queries, err := um.GenerateMemoryQueries(ctx, query)
	if err != nil {
		errnie.Info(fmt.Sprintf("Failed to generate memory queries: %v", err))
		queries = []string{query} // Fall back to the original query
	}

	// Retrieve memories for each query
	var allMemories []EnhancedMemoryEntry
	seenMemoryIDs := make(map[string]bool)
	maxMemoriesPerQuery := um.options.MaxMemoriesPerQuery / len(queries)
	if maxMemoriesPerQuery < 1 {
		maxMemoriesPerQuery = 1
	}

	for _, q := range queries {
		memories, err := um.RetrieveMemoriesByVector(
			ctx,
			q,
			agentID,
			maxMemoriesPerQuery,
			um.options.ExtractionThreshold,
		)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to retrieve memories for query '%s': %v", q, err))
			continue
		}

		// Add new memories to the result list, avoiding duplicates
		for _, memory := range memories {
			if !seenMemoryIDs[memory.ID] {
				allMemories = append(allMemories, memory)
				seenMemoryIDs[memory.ID] = true
			}
		}
	}

	// If no memories, just return the query
	if len(allMemories) == 0 {
		return query, nil
	}

	// Sort memories by relevance (assuming most recent queries produced most relevant results)
	// For a real implementation, you might want to score memories differently

	// Format memories as text
	var memoriesText strings.Builder
	for i, memory := range allMemories {
		memoriesText.WriteString(fmt.Sprintf("%d. %s\n", i+1, memory.Content))
	}

	// Use the template to format the context
	data := struct {
		Memories string
		Query    string
	}{
		Memories: memoriesText.String(),
		Query:    query,
	}

	var result strings.Builder
	err = um.contextTemplate.Execute(&result, data)
	if err != nil {
		errnie.Info(fmt.Sprintf("Failed to format context with memories: %v", err))
		return query, nil
	}

	return result.String(), nil
}

/*
SummarizeMemories generates a summary of a collection of memories.
It creates a textual summary of the provided memory entries.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - entries: The memory entries to summarize

Returns:
  - A textual summary of the memories
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) SummarizeMemories(ctx context.Context, entries []EnhancedMemoryEntry) (string, error) {
	// This would normally use an LLM to summarize the memories
	// For now, just concatenate them
	var summary strings.Builder
	summary.WriteString("Memory Summary:\n")

	for i, entry := range entries {
		summary.WriteString(fmt.Sprintf("%d. %s\n", i+1, entry.Content))
	}

	return summary.String(), nil
}
