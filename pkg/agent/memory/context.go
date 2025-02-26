package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/util"
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
func (um *UnifiedMemory) StoreMemory(
	ctx context.Context,
	agentID string,
	content string,
	memType MemoryType,
	source string,
	metadata map[string]string,
) (string, error) {
	// Generate a unique ID for the memory
	memoryID := uuid.New().String()

	// Set defaults for metadata
	if metadata == nil {
		metadata = make(map[string]string)
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
func (um *UnifiedMemory) ExtractMemories(
	ctx context.Context,
	agentID string,
	content string,
	source string,
) ([]string, error) {
	provider := llm.NewOpenAIProvider(
		os.Getenv("OPENAI_API_KEY"),
		"gpt-4o-mini",
	)

	schema := util.GenerateSchema[MemoryExtraction]()

	res := provider.GenerateResponse(ctx, core.LLMParams{
		Messages: []core.LLMMessage{
			{
				Role:    "system",
				Content: viper.GetViper().GetString("templates.memory"),
			},
			{
				Role:    "user",
				Content: fmt.Sprintf("Text to extract memories from: %s", content),
			},
		},
		ResponseFormatName:        "memory_extraction",
		ResponseFormatDescription: "A JSON object with a 'documents' array of strings",
		Schema:                    schema,
	})

	// Parse the response directly - should be valid JSON with the schema
	var extraction MemoryExtraction
	err := json.Unmarshal([]byte(res.Content), &extraction)

	if err != nil {
		errnie.Error(fmt.Errorf("failed to parse memory extraction JSON: %w", err))
		return []string{}, err
	}

	for _, doc := range extraction.Documents {
		um.StoreDocumentMemory(ctx, agentID, doc.Document, MemoryTypeGlobal, source, doc.Metadata)
	}

	for _, entity := range extraction.Entities {
		props := map[string]interface{}{
			"source":     source,
			"created_at": time.Now().Format(time.RFC3339),
		}

		for k, v := range entity.Metadata {
			props[k] = v
		}

		err := um.graphStore.Cypher(ctx, `
		CALL apoc.merge.node([$entityLabel], {name: $entityName}) YIELD node AS n
		CALL apoc.merge.node([$targetLabel], {name: $targetName}) YIELD node AS m
		CALL apoc.create.relationship(n, $relationshipType, $props, m) YIELD rel
		RETURN n, rel, m`, map[string]interface{}{
			"entityLabel":      entity.Entity,
			"entityName":       entity.Entity, // Assuming unique name, adjust as needed
			"targetLabel":      entity.Target,
			"targetName":       entity.Target, // Assuming unique name, adjust as needed
			"relationshipType": entity.Relationship,
			"props": map[string]interface{}{
				"source":     source,
				"created_at": time.Now().Format(time.RFC3339),
			},
		})

		if err != nil {
			log.Printf("Error executing Cypher query: %v", err)
		}
	}

	return nil, nil
}

// StoreDocumentMemory stores a document memory in vector store only
func (um *UnifiedMemory) StoreDocumentMemory(
	ctx context.Context,
	agentID string,
	content string,
	memType MemoryType,
	source string,
	metadata map[string]string,
) (string, error) {
	// Generate a unique ID for the memory
	memoryID := uuid.New().String()

	// Set defaults for metadata
	if metadata == nil {
		metadata = make(map[string]string)
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

	return memoryID, nil
}

// findEntityNodeByContent looks for an existing node with the given content
func (um *UnifiedMemory) findEntityNodeByContent(ctx context.Context, content string) (string, error) {
	if um.graphStore == nil {
		return "", fmt.Errorf("graph store not available")
	}

	// Debug the query we're about to run
	errnie.Debug(fmt.Sprintf("Searching for entity with content: %s", content))

	query := `
		MATCH (n:Entity)
		WHERE n.content = $content
		RETURN n.id as id
	`
	params := map[string]interface{}{
		"content": content,
	}

	results, err := um.graphStore.Query(ctx, query, params)
	if err != nil {
		return "", err
	}

	if len(results) == 0 {
		return "", fmt.Errorf("node not found")
	}

	id, ok := results[0]["id"].(string)
	if !ok {
		return "", fmt.Errorf("invalid ID format")
	}

	return id, nil
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

	// Get the system prompt from config
	systemPrompt := viper.GetViper().GetString("templates.memory_query")
	if systemPrompt == "" {
		errnie.Info("Memory query template not found in config, using default")
		systemPrompt = "You are a memory query generator for an AI assistant. Your job is to analyze the user's query and generate 3-5 search queries that will help retrieve relevant information from the assistant's memory. The queries should be diverse to cover different aspects of what might be relevant. Each query should be a short phrase or question that captures an important aspect of the information needed."
	}
	userPrompt := fmt.Sprintf("Generate search queries for: %s", query)

	// Create an OpenAI provider
	provider := llm.NewOpenAIProvider(
		os.Getenv("OPENAI_API_KEY"),
		"gpt-4o-mini",
	)

	schema := util.GenerateSchema[MemoryQueries]()

	// Call the OpenAI API using the provider
	res := provider.GenerateResponse(ctx, core.LLMParams{
		Messages: []core.LLMMessage{
			{
				Role:    "system",
				Content: systemPrompt,
			},
			{
				Role:    "user",
				Content: userPrompt,
			},
		},
		ResponseFormatName:        "memory_queries",
		ResponseFormatDescription: "A JSON object with a 'queries' array of strings",
		Schema:                    schema,
	})

	// Log the raw response for debugging
	errnie.Debug("Raw LLM Memory Query Response:", res.Content)

	// Parse the response directly as JSON
	var queriesResponse MemoryQueries
	err := json.Unmarshal([]byte(res.Content), &queriesResponse)

	if err != nil {
		errnie.Error(fmt.Errorf("failed to parse memory queries JSON: %w", err))
		return []string{query}, nil
	}

	// Validate and process the queries
	var validQueries []string

	for _, q := range queriesResponse.Queries {
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
	maxMemoriesPerQuery := max(um.options.MaxMemoriesPerQuery/len(queries), 1)

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
