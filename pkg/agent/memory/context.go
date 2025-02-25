package memory

import (
	"context"
	"encoding/json"
	"fmt"
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

// getOpenAIProvider creates and returns an OpenAI provider with the specified model
// This helper function can be reused across memory functions
func getOpenAIProvider(model string) (*llm.OpenAIProvider, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}
	return llm.NewOpenAIProvider(apiKey, model), nil
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
	provider, err := getOpenAIProvider("gpt-4o-mini")

	if err != nil {
		errnie.Error(err)
		return []string{}, err
	}

	// Call the OpenAI API using the provider
	extractedContent, err := provider.GenerateResponse(ctx, core.LLMParams{
		Messages: []core.LLMMessage{
			{
				Role:    "system",
				Content: viper.GetViper().GetString("templates.memory"),
			},
			{
				Role:    "user",
				Content: fmt.Sprintf("Text to extract memories from: %s", text),
			},
		},
		ResponseFormatName:        "json_schema",
		ResponseFormatDescription: "A JSON object with a 'documents' array of strings",
		Schema:                    util.GenerateSchema[MemoryExtraction](),
	})

	if err != nil {
		errnie.Error(err)
		return []string{}, err
	}

	// Log the raw response for debugging
	errnie.Debug(fmt.Sprintf("Raw LLM response: %s", extractedContent))

	// Parse the response directly - should be valid JSON with the schema
	var extraction MemoryExtraction
	err = json.Unmarshal([]byte(extractedContent), &extraction)
	if err != nil {
		errnie.Error(fmt.Errorf("failed to parse memory extraction JSON: %w", err))
		return []string{}, err
	}

	// Validate extraction results
	if len(extraction.Documents) == 0 {
		errnie.Info("No document memories found in extraction results")
		if len(extraction.Entities) == 0 {
			errnie.Info("No entities found either")
		}
	} else {
		errnie.Debug(fmt.Sprintf("Found %d documents and %d entities in extraction results",
			len(extraction.Documents), len(extraction.Entities)))
	}

	// Store document memories in vector store
	var memoryIDs []string
	documentToID := make(map[string]string) // Map document content to its ID for relationship creation

	// Process documents
	for _, doc := range extraction.Documents {
		if doc.Document == "" || len(doc.Document) < 30 {
			continue // Skip empty or very short documents
		}

		// Apply metadata if it exists or initialize empty
		metadata := doc.Metadata
		if metadata == nil {
			metadata = make(map[string]interface{})
		}

		// Ensure source is set
		if _, exists := metadata["source"]; !exists {
			metadata["source"] = source
		}

		// Store the document memory
		memoryID, err := um.StoreMemory(ctx, agentID, doc.Document, MemoryTypePersonal, source, metadata)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to store document memory: %v", err))
			continue
		}
		memoryIDs = append(memoryIDs, memoryID)
		documentToID[doc.Document] = memoryID
	}

	// Process entity relationships if graph store is available
	if um.options.EnableGraphStore && um.graphStore != nil && len(extraction.Entities) > 0 {
		// Process each entity relationship
		for _, entity := range extraction.Entities {
			if entity.Entity == "" || entity.Relationship == "" {
				continue // Skip incomplete entities
			}

			// Apply metadata if it exists or initialize empty
			metadata := entity.Metadata
			if metadata == nil {
				metadata = make(map[string]interface{})
			}

			// Ensure source is set
			if _, exists := metadata["source"]; !exists {
				metadata["source"] = source
			}

			// Get target entity directly now that we have proper schema
			targetText := entity.Target
			if targetText == "" {
				errnie.Info(fmt.Sprintf("Entity '%s' has no relationship target", entity.Entity))
				continue
			}

			// Create entity node if it doesn't already have an ID
			entityID, exists := documentToID[entity.Entity]
			if !exists {
				// Store entity as a new memory
				entityID, err = um.StoreMemory(ctx, agentID, entity.Entity, MemoryTypePersonal, source, metadata)
				if err != nil {
					errnie.Info(fmt.Sprintf("Failed to store entity: %v", err))
					continue
				}
				memoryIDs = append(memoryIDs, entityID)
				documentToID[entity.Entity] = entityID
			}

			// Get or create target entity
			targetID, exists := documentToID[targetText]
			if !exists {
				// Store target as a new memory
				targetID, err = um.StoreMemory(ctx, agentID, targetText, MemoryTypePersonal, source, metadata)
				if err != nil {
					errnie.Info(fmt.Sprintf("Failed to store target entity: %v", err))
					continue
				}
				memoryIDs = append(memoryIDs, targetID)
				documentToID[targetText] = targetID
			}

			// Create relationship properties
			relationProps := map[string]interface{}{
				"created": time.Now().Format(time.RFC3339),
				"source":  source,
			}

			// Add any metadata from the entity to the relationship
			for k, v := range metadata {
				relationProps[k] = v
			}

			// Create the relationship in the graph store
			if err := um.graphStore.CreateRelationship(ctx, entityID, targetID, entity.Relationship, relationProps); err != nil {
				errnie.Info(fmt.Sprintf("Failed to create relationship: %v", err))
			}
		}
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

	// Get the system prompt from config
	systemPrompt := viper.GetViper().GetString("templates.memory_query")
	if systemPrompt == "" {
		errnie.Info("Memory query template not found in config, using default")
		systemPrompt = "You are a memory query generator for an AI assistant. Your job is to analyze the user's query and generate 3-5 search queries that will help retrieve relevant information from the assistant's memory. The queries should be diverse to cover different aspects of what might be relevant. Each query should be a short phrase or question that captures an important aspect of the information needed."
	}
	userPrompt := fmt.Sprintf("Generate search queries for: %s", query)

	// Create an OpenAI provider
	provider, err := getOpenAIProvider("gpt-4o-mini")
	if err != nil {
		errnie.Error(err)
		return []string{query}, nil
	}

	// Call the OpenAI API using the provider
	extractedContent, err := provider.GenerateResponse(ctx, core.LLMParams{
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
		ResponseFormatName:        "json_schema",
		ResponseFormatDescription: "A JSON object with a 'queries' array of strings",
		Schema:                    util.GenerateSchema[MemoryQueries](),
	})

	if err != nil {
		errnie.Error(fmt.Errorf("failed to generate memory queries: %w", err))
		return []string{query}, nil
	}

	// Log the raw response for debugging
	errnie.Debug("Raw LLM Memory Query Response:", extractedContent)

	// Parse the response directly as JSON
	var queriesResponse MemoryQueries
	err = json.Unmarshal([]byte(extractedContent), &queriesResponse)
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
