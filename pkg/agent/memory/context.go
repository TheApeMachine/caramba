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
	"github.com/theapemachine/caramba/pkg/output"
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
	output.Verbose(fmt.Sprintf("Storing memory for agent %s (%d chars)", agentID, len(content)))

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
		embeddingSpinner := output.StartSpinner("Generating memory embedding")
		embedding, err = um.embeddingProvider.GetEmbedding(ctx, content)
		if err != nil {
			output.StopSpinner(embeddingSpinner, "")
			output.Error("Failed to get embedding for memory", err)
			errnie.Info(fmt.Sprintf("Failed to get embedding for memory: %v", err))
		} else {
			output.StopSpinner(embeddingSpinner, fmt.Sprintf("Generated embedding vector (%d dimensions)", len(embedding)))
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
		vectorStoreSpinner := output.StartSpinner("Storing in vector database")

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
			output.StopSpinner(vectorStoreSpinner, "")
			output.Error("Failed to store memory in vector store", err)
			errnie.Info(fmt.Sprintf("Failed to store memory in vector store: %v", err))
		} else {
			output.StopSpinner(vectorStoreSpinner, "Memory stored in vector database")
		}
	}

	// Store in graph store if available
	if um.options.EnableGraphStore && um.graphStore != nil {
		graphStoreSpinner := output.StartSpinner("Storing in graph database")

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
			output.StopSpinner(graphStoreSpinner, "")
			output.Error("Failed to store memory in graph store", err)
			errnie.Info(fmt.Sprintf("Failed to store memory in graph store: %v", err))
		} else {
			output.StopSpinner(graphStoreSpinner, "Memory stored in graph database")
		}
	}

	// For simplicity, also store in the base memory store
	err = um.baseStore.Store(ctx, "memory:"+memoryID, content)
	if err != nil {
		output.Error("Failed to store memory in base store", err)
		errnie.Info(fmt.Sprintf("Failed to store memory in base store: %v", err))
	}

	output.Result(fmt.Sprintf("Memory stored with ID: %s", memoryID))
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
	output.Action("memory", "extract", fmt.Sprintf("Extracting memories from %s", source))

	// Create spinner for extraction process
	extractSpinner := output.StartSpinner("Analyzing content for important information")

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

	output.Debug("Raw memory extraction response: " + output.Summarize(res.Content, 80))

	// Parse the response directly - should be valid JSON with the schema
	var extraction MemoryExtraction
	err := json.Unmarshal([]byte(res.Content), &extraction)

	if err != nil {
		output.StopSpinner(extractSpinner, "")
		output.Error("Memory extraction failed", err)
		errnie.Error(fmt.Errorf("failed to parse memory extraction JSON: %w", err))
		return []string{}, err
	}

	// Stop spinner and show extraction results
	output.StopSpinner(extractSpinner, fmt.Sprintf("Extracted %d documents and %d entity relationships",
		len(extraction.Documents), len(extraction.Entities)))

	// Store document memories
	docResults := make([]string, 0, len(extraction.Documents))
	for _, doc := range extraction.Documents {
		memID, err := um.StoreDocumentMemory(ctx, agentID, doc.Document, MemoryTypeGlobal, source, doc.Metadata)
		if err != nil {
			output.Error("Failed to store document memory", err)
		} else {
			docResults = append(docResults, memID)
			output.Verbose(fmt.Sprintf("Stored document memory: %s", output.Summarize(doc.Document, 40)))
		}
	}

	// Store entity relationships
	relationshipSpinner := output.StartSpinner("Processing entity relationships")
	relationshipCount := 0

	for _, entity := range extraction.Entities {
		props := map[string]interface{}{
			"source":     source,
			"created_at": time.Now().Format(time.RFC3339),
		}

		for k, v := range entity.Metadata {
			props[k] = v
		}

		// Less verbose Cypher query logging for relationship creation
		output.Debug(fmt.Sprintf("Creating relationship: %s -> %s -> %s",
			entity.Entity, entity.Relationship, entity.Target))

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
			output.Debug(fmt.Sprintf("Error creating relationship: %v", err))
			log.Printf("Error executing Cypher query: %v", err)
		} else {
			relationshipCount++
		}
	}

	output.StopSpinner(relationshipSpinner, fmt.Sprintf("Created %d entity relationships", relationshipCount))

	return docResults, nil
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
			output.Debug(fmt.Sprintf("Failed to get embedding for document memory: %v", err))
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
			output.Debug(fmt.Sprintf("Failed to store document memory in vector store: %v", err))
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
	output.Debug(fmt.Sprintf("Searching for entity with content: %s", output.Summarize(content, 30)))

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
	output.Verbose(fmt.Sprintf("Generating memory queries for: %s", output.Summarize(query, 40)))

	// Avoid API calls for very short queries
	if len(query) < 10 {
		output.Debug("Query too short, using as-is")
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
	output.Debug("Raw Memory Query Response: " + output.Summarize(res.Content, 80))

	// Parse the response directly as JSON
	var queriesResponse MemoryQueries
	err := json.Unmarshal([]byte(res.Content), &queriesResponse)

	if err != nil {
		output.Error("Failed to parse memory queries", err)
		errnie.Error(fmt.Errorf("failed to parse memory queries JSON: %w", err))
		return []string{query}, nil
	}

	// Validate and process the queries
	var validQueries []string

	for _, q := range queriesResponse.Queries {
		q = strings.TrimSpace(q)
		if q != "" {
			validQueries = append(validQueries, q)
			output.Debug("Generated query: " + q)
		}
	}

	output.Result(fmt.Sprintf("Generated %d memory search queries", len(validQueries)))

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
	output.Action("memory", "prepare_context", fmt.Sprintf("Enhancing context for agent %s", agentID))

	// Default to just returning the query if no vector store
	if !um.options.EnableVectorStore || um.vectorStore == nil {
		output.Info("Vector store is disabled or not available")
		return query, nil
	}

	// Generate memory queries using LLM
	memQuerySpinner := output.StartSpinner("Generating memory search queries")
	queries, err := um.GenerateMemoryQueries(ctx, query)

	// Detailed logging of the generated queries
	if err == nil && len(queries) > 0 {
		output.Verbose("Generated the following memory search queries:")
		for i, q := range queries {
			output.Verbose(fmt.Sprintf("  Query %d: %s", i+1, q))
		}
	}

	output.StopSpinner(memQuerySpinner, fmt.Sprintf("Generated %d memory search queries", len(queries)))

	if err != nil {
		output.Error("Failed to generate memory queries", err)
		errnie.Info(fmt.Sprintf("Failed to generate memory queries: %v", err))
		queries = []string{query} // Fall back to the original query
	}

	// Retrieve memories for each query
	var allMemories []EnhancedMemoryEntry
	seenMemoryIDs := make(map[string]bool)
	maxMemoriesPerQuery := max(um.options.MaxMemoriesPerQuery/len(queries), 1)

	memRetrieveSpinner := output.StartSpinner("Retrieving relevant memories")

	totalRetrieved := 0
	for i, q := range queries {
		output.Verbose(fmt.Sprintf("Searching memory with query %d: %s", i+1, q))

		memories, err := um.RetrieveMemoriesByVector(
			ctx,
			q,
			agentID,
			maxMemoriesPerQuery,
			um.options.ExtractionThreshold,
		)
		if err != nil {
			output.Debug(fmt.Sprintf("Failed to retrieve memories for query '%s': %v", q, err))
			errnie.Info(fmt.Sprintf("Failed to retrieve memories for query '%s': %v", q, err))
			continue
		}

		totalRetrieved += len(memories)
		output.Verbose(fmt.Sprintf("Query %d returned %d memories", i+1, len(memories)))

		// Add new memories to the result list, avoiding duplicates
		for _, memory := range memories {
			if !seenMemoryIDs[memory.ID] {
				allMemories = append(allMemories, memory)
				seenMemoryIDs[memory.ID] = true
				output.Debug(fmt.Sprintf("Adding memory: %s", output.Summarize(memory.Content, 40)))
			} else {
				output.Debug(fmt.Sprintf("Skipping duplicate memory: %s", output.Summarize(memory.Content, 40)))
			}
		}
	}

	output.StopSpinner(memRetrieveSpinner, fmt.Sprintf("Retrieved %d unique memories", len(allMemories)))

	// If no memories, just return the query
	if len(allMemories) == 0 {
		output.Info("No relevant memories found for this query")
		return query, nil
	}

	// Sort memories by relevance (assuming most recent queries produced most relevant results)
	// For a real implementation, you might want to score memories differently

	// Format memories as text
	var memoriesText strings.Builder
	output.Info("Including the following memories in context:")
	for i, memory := range allMemories {
		memoryLine := fmt.Sprintf("%d. %s\n", i+1, memory.Content)
		memoriesText.WriteString(memoryLine)
		output.Verbose(fmt.Sprintf("Memory %d: %s", i+1, output.Summarize(memory.Content, 60)))
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
		output.Error("Failed to format context with memories", err)
		errnie.Info(fmt.Sprintf("Failed to format context with memories: %v", err))
		return query, nil
	}

	enhancedContext := result.String()
	originalLength := len(query)
	enhancedLength := len(enhancedContext)
	addedChars := enhancedLength - originalLength

	output.Result(fmt.Sprintf("Enhanced context with %d memories (added %d characters)", len(allMemories), addedChars))
	output.Debug(fmt.Sprintf("Original query length: %d chars, Enhanced context: %d chars", originalLength, enhancedLength))

	// If we didn't actually add any memories (template issue?)
	if addedChars <= 0 {
		output.Warn("Memory enhancement didn't add any content - check context template")
		return query, nil
	}

	return enhancedContext, nil
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
	output.Action("memory", "summarize", fmt.Sprintf("Summarizing %d memories", len(entries)))

	// This would normally use an LLM to summarize the memories
	// For now, just concatenate them
	var summary strings.Builder
	summary.WriteString("Memory Summary:\n")

	for i, entry := range entries {
		summary.WriteString(fmt.Sprintf("%d. %s\n", i+1, entry.Content))
	}

	output.Result("Memory summary generated")
	return summary.String(), nil
}
