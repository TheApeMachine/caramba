package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
It identifies key facts, concepts, and insights from the text and stores them as memories.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - agentID: The ID of the agent extracting the memories
  - text: The text to extract memories from
  - source: The source of the text (e.g., "conversation", "document", etc.)

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

	// Create a system prompt for memory extraction that encourages document-like extraction
	// and identifies relationships between memories
	systemPrompt := `You are an advanced memory extraction system for an AI agent. Extract important information from the provided text that should be stored as memories.

TASK 1: Extract 2-5 substantial, document-like memories that would be valuable to remember later.
- Each memory should be 2-4 sentences long (50-200 words), forming a coherent paragraph
- Focus on factual information, key insights, and important details
- Include context and supporting details to make each memory self-contained and useful
- Format each memory as a complete, standalone paragraph
- Separate memories with exactly two newlines

TASK 2: After extracting the memories, identify relationships between them in this format:
RELATIONSHIPS:
memory1 -> RELATES_TO -> memory2: brief description of how they relate
memory2 -> PART_OF -> memory3: brief description of the part-whole relationship
memory1 -> CAUSES -> memory3: brief description of the causal relationship

Use relationship types like: RELATES_TO, PART_OF, CAUSES, FOLLOWS, CONTRADICTS, SUPPORTS, EXAMPLE_OF
Don't create relationships if they don't naturally exist.`

	userPrompt := fmt.Sprintf("Text to extract memories from: %s", text)

	// Create the request
	reqBody := OpenAIRequest{
		Model: "gpt-3.5-turbo",
		Messages: []Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		MaxTokens: 800, // Increased from 500 to allow for more detailed memories and relationships
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

	// Split the content into memories and relationships sections
	sections := strings.Split(extractedContent, "RELATIONSHIPS:")

	// Process the memories section
	var documentMemories []string
	if len(sections) > 0 && len(sections[0]) > 0 {
		// Split memories by double newlines as requested in the prompt
		memories := strings.Split(strings.TrimSpace(sections[0]), "\n\n")
		for _, memory := range memories {
			memory = strings.TrimSpace(memory)
			if memory == "" || len(memory) < 30 { // Skip very short entries
				continue
			}
			documentMemories = append(documentMemories, memory)
		}
	}

	// Process and store the document-like memories
	var memoryIDs []string
	memoryContentToID := make(map[string]string) // Map memory content to its ID for relationship creation

	for _, memory := range documentMemories {
		// Store the memory as a document-like chunk
		memoryID, err := um.StoreMemory(ctx, agentID, memory, MemoryTypePersonal, source, nil)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to store extracted memory: %v", err))
			continue
		}
		memoryIDs = append(memoryIDs, memoryID)
		memoryContentToID[memory] = memoryID
	}

	// Process relationships if they exist and graph store is enabled
	if len(sections) > 1 && um.options.EnableGraphStore && um.graphStore != nil {
		relationshipsText := strings.TrimSpace(sections[1])
		if relationshipsText != "" {
			// Parse and create relationships
			relationships := strings.Split(relationshipsText, "\n")
			for _, rel := range relationships {
				rel = strings.TrimSpace(rel)
				if rel == "" {
					continue
				}

				// Try to parse the relationship
				if err := um.processRelationship(ctx, rel, memoryContentToID); err != nil {
					errnie.Info(fmt.Sprintf("Failed to process relationship: %v", err))
				}
			}
		}
	}

	return memoryIDs, nil
}

// processRelationship parses a relationship string and creates the relationship in the graph store
// Format expected: memory1 -> RELATES_TO -> memory2: description
func (um *UnifiedMemory) processRelationship(ctx context.Context, relationshipText string, memoryContentToID map[string]string) error {
	// Skip if graph store is not enabled
	if !um.options.EnableGraphStore || um.graphStore == nil {
		return errors.New("graph store not enabled")
	}

	// Parse the relationship format
	parts := strings.Split(relationshipText, "->")
	if len(parts) < 3 {
		return fmt.Errorf("invalid relationship format: %s", relationshipText)
	}

	// Extract source, relationship type, and target
	sourceMemoryHint := strings.TrimSpace(parts[0])
	relType := strings.TrimSpace(parts[1])

	// The last part contains both the target and description, split by colon
	targetAndDesc := parts[2]
	targetDescParts := strings.SplitN(targetAndDesc, ":", 2)

	var targetMemoryHint, description string
	if len(targetDescParts) > 1 {
		// We have both target and description
		targetMemoryHint = strings.TrimSpace(targetDescParts[0])
		description = strings.TrimSpace(targetDescParts[1])
	} else {
		// Only target, no description
		targetMemoryHint = strings.TrimSpace(targetDescParts[0])
		description = "Related memory"
	}

	// Find the memory IDs based on the hints (partial content match)
	var sourceID, targetID string

	// Find source memory ID
	for content, id := range memoryContentToID {
		if strings.Contains(content, sourceMemoryHint) || strings.Contains(sourceMemoryHint, content) {
			sourceID = id
			break
		}
	}

	// Find target memory ID
	for content, id := range memoryContentToID {
		if strings.Contains(content, targetMemoryHint) || strings.Contains(targetMemoryHint, content) {
			targetID = id
			break
		}
	}

	if sourceID == "" || targetID == "" {
		return fmt.Errorf("could not identify memory IDs for relationship: %s", relationshipText)
	}

	// Create the relationship in the graph store
	properties := map[string]interface{}{
		"description": description,
		"created":     time.Now().Format(time.RFC3339),
		"source":      "memory_extraction",
	}

	return um.graphStore.CreateRelationship(ctx, sourceID, targetID, relType, properties)
}

/*
basicExtractMemories provides a fallback extraction method when LLM is unavailable.
It extracts memories using simple heuristics rather than LLM-based extraction.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - agentID: The ID of the agent extracting the memories
  - text: The text to extract memories from
  - source: The source of the text (e.g., "conversation", "document", etc.)

Returns:
  - A slice of memory IDs that were created
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) basicExtractMemories(ctx context.Context, agentID string, text string, source string) ([]string, error) {
	// Split text into paragraphs
	paragraphs := strings.Split(text, "\n\n")
	var documentMemories []string
	var memoryIDs []string

	// First pass: collect substantial paragraphs as document-like memories
	for _, paragraph := range paragraphs {
		paragraph = strings.TrimSpace(paragraph)
		// Skip very short paragraphs, but keep moderate to long ones as document chunks
		if len(paragraph) < 50 {
			continue
		}

		// If paragraph is too long, break it down
		if len(paragraph) > 500 {
			// Break into roughly equal parts
			sentences := strings.Split(paragraph, ".")
			var currentChunk string
			for _, sentence := range sentences {
				sentence = strings.TrimSpace(sentence)
				if sentence == "" {
					continue
				}

				// Add period back to sentence
				sentence += "."

				// If adding this sentence would make chunk too long, save current chunk and start new one
				if len(currentChunk)+len(sentence) > 300 && len(currentChunk) > 100 {
					documentMemories = append(documentMemories, strings.TrimSpace(currentChunk))
					currentChunk = sentence
				} else {
					currentChunk += " " + sentence
				}
			}

			// Don't forget the last chunk
			if len(currentChunk) > 50 {
				documentMemories = append(documentMemories, strings.TrimSpace(currentChunk))
			}
		} else {
			// Paragraph is a good size for a document memory
			documentMemories = append(documentMemories, paragraph)
		}
	}

	// Second pass: if we don't have enough document memories, look for significant sentences
	if len(documentMemories) < 2 {
		// Process all text to find significant sentences
		sentences := extractSignificantSentences(text)

		// Combine sentences into coherent chunks of 2-4 sentences
		var currentChunk string
		sentenceCount := 0

		for _, sentence := range sentences {
			sentenceCount++
			if currentChunk == "" {
				currentChunk = sentence
			} else {
				currentChunk += " " + sentence
			}

			// After 2-4 sentences, create a document memory
			if sentenceCount >= 2 && (sentenceCount >= 4 || len(currentChunk) > 150) {
				documentMemories = append(documentMemories, currentChunk)
				currentChunk = ""
				sentenceCount = 0
			}
		}

		// Add any remaining sentences
		if currentChunk != "" {
			documentMemories = append(documentMemories, currentChunk)
		}
	}

	// Limit the number of memories to extract
	maxMemories := 5
	if len(documentMemories) > maxMemories {
		documentMemories = documentMemories[:maxMemories]
	}

	// Store the memories
	memoryContentToID := make(map[string]string)
	for _, memory := range documentMemories {
		memoryID, err := um.StoreMemory(ctx, agentID, memory, MemoryTypePersonal, source, nil)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to store extracted memory: %v", err))
			continue
		}
		memoryIDs = append(memoryIDs, memoryID)
		memoryContentToID[memory] = memoryID
	}

	// If we have graph store enabled and multiple memories, create basic relationships
	if um.options.EnableGraphStore && um.graphStore != nil && len(memoryIDs) > 1 {
		// Create sequential relationships between memories (as a fallback approach)
		for i := 0; i < len(memoryIDs)-1; i++ {
			properties := map[string]interface{}{
				"description": "Sequential memory",
				"created":     time.Now().Format(time.RFC3339),
				"source":      "basic_extraction",
			}

			// Create a FOLLOWS relationship between consecutive memories
			err := um.graphStore.CreateRelationship(ctx, memoryIDs[i], memoryIDs[i+1], "FOLLOWS", properties)
			if err != nil {
				errnie.Info(fmt.Sprintf("Failed to create sequential relationship: %v", err))
			}
		}

		// Create a PART_OF relationship to a common group for all memories from this extraction
		groupID := fmt.Sprintf("group_%d", time.Now().UnixNano())
		labels := []string{"MemoryGroup", source}
		properties := map[string]interface{}{
			"created": time.Now().Format(time.RFC3339),
			"source":  source,
		}

		// Create the group node
		if err := um.graphStore.CreateNode(ctx, groupID, labels, properties); err == nil {
			// Connect each memory to the group
			for _, memoryID := range memoryIDs {
				relationProperties := map[string]interface{}{
					"description": "Memory belongs to this extraction group",
					"created":     time.Now().Format(time.RFC3339),
				}

				_ = um.graphStore.CreateRelationship(ctx, memoryID, groupID, "PART_OF", relationProperties)
			}
		}
	}

	return memoryIDs, nil
}

// extractSignificantSentences finds sentences that are likely to be important
func extractSignificantSentences(text string) []string {
	var significantSentences []string

	// Split into sentences
	sentences := strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?'
	})

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
			strings.Contains(lowerSentence, "note that") ||
			strings.Contains(lowerSentence, "remember") ||
			strings.Contains(lowerSentence, "crucial") ||
			strings.Contains(lowerSentence, "main") ||
			strings.Contains(lowerSentence, "fundamental") {
			significantSentences = append(significantSentences, sentence)
		}
	}

	// If we don't have enough significant sentences, take a sampling of regular sentences
	if len(significantSentences) < 3 {
		for _, sentence := range sentences {
			sentence = strings.TrimSpace(sentence)
			if len(sentence) >= 40 && len(sentence) <= 200 {
				significantSentences = append(significantSentences, sentence)
				if len(significantSentences) >= 5 {
					break
				}
			}
		}
	}

	return significantSentences
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
