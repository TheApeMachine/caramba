/*
Package memory provides various memory storage implementations for Caramba agents.
This package offers different memory storage solutions, from simple in-memory
stores to sophisticated vector and graph-based memory implementations that
support semantic search and complex relationships between memory items.
*/
package memory

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"text/template"
	"time"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/errnie"
)

/*
MemoryType represents the type of memory (personal or global).
It distinguishes between agent-specific memories and shared global memories.
*/
type MemoryType string

const (
	/* MemoryTypePersonal represents an agent's personal memory */
	MemoryTypePersonal MemoryType = "personal"

	/* MemoryTypeGlobal represents shared global memory */
	MemoryTypeGlobal MemoryType = "global"
)

/*
MemoryStoreType represents the backend memory store type.
It identifies the storage implementation used for memories.
*/
type MemoryStoreType string

const (
	/* MemoryStoreVector represents vector-based memory storage (e.g., QDrant) */
	MemoryStoreVector MemoryStoreType = "vector"

	/* MemoryStoreGraph represents graph-based memory storage (e.g., Neo4j) */
	MemoryStoreGraph MemoryStoreType = "graph"
)

/*
EnhancedMemoryEntry represents a single memory entry with enhanced metadata.
It extends the basic memory entry with additional metadata, embedding information,
and tracking data for more sophisticated memory management.
*/
type EnhancedMemoryEntry struct {
	/* ID is a unique identifier for the memory */
	ID string
	/* AgentID is the ID of the agent who owns the memory (empty for global) */
	AgentID string
	/* Content is the actual memory content */
	Content string
	/* Embedding is the vector embedding of the content */
	Embedding []float32
	/* Type indicates the type of memory (personal/global) */
	Type MemoryType
	/* Source identifies the origin of the memory (conversation, document, etc.) */
	Source string
	/* CreatedAt records when the memory was created */
	CreatedAt time.Time
	/* AccessCount tracks how many times this memory has been accessed */
	AccessCount int
	/* LastAccess records when this memory was last accessed */
	LastAccess time.Time
	/* Metadata stores additional information about the memory */
	Metadata map[string]interface{}
}

/*
ToMemoryEntry converts an EnhancedMemoryEntry to a standard core.MemoryEntry.
This method simplifies the enhanced entry into the basic format used by the core memory interface.

Returns:
  - A core.MemoryEntry representation of this enhanced entry
*/
func (e *EnhancedMemoryEntry) ToMemoryEntry() core.MemoryEntry {
	return core.MemoryEntry{
		Key:   e.ID,
		Value: e.Content,
		Score: 1.0, // Default score
	}
}

/*
FromCoreEnhancedEntry converts from core.EnhancedMemoryEntry to local EnhancedMemoryEntry.
This function bridges between the core package's memory types and the memory package's types.

Parameters:
  - entry: The core package's enhanced memory entry to convert

Returns:
  - The converted local EnhancedMemoryEntry
*/
func FromCoreEnhancedEntry(entry core.EnhancedMemoryEntry) EnhancedMemoryEntry {
	return EnhancedMemoryEntry{
		ID:          entry.ID,
		AgentID:     entry.AgentID,
		Content:     entry.Content,
		Embedding:   entry.Embedding,
		Type:        MemoryType(entry.Type),
		Source:      entry.Source,
		CreatedAt:   entry.CreatedAt,
		AccessCount: entry.AccessCount,
		LastAccess:  entry.LastAccess,
		Metadata:    entry.Metadata,
	}
}

/*
ToCoreEnhancedEntry converts to core.EnhancedMemoryEntry from local EnhancedMemoryEntry.
This method bridges between the memory package's types and the core package's memory types.

Returns:
  - The converted core.EnhancedMemoryEntry
*/
func (e *EnhancedMemoryEntry) ToCoreEnhancedEntry() core.EnhancedMemoryEntry {
	return core.EnhancedMemoryEntry{
		ID:          e.ID,
		AgentID:     e.AgentID,
		Content:     e.Content,
		Embedding:   e.Embedding,
		Type:        core.MemoryType(e.Type),
		Source:      e.Source,
		CreatedAt:   e.CreatedAt,
		AccessCount: e.AccessCount,
		LastAccess:  e.LastAccess,
		Metadata:    e.Metadata,
	}
}

/*
Relationship represents a relationship between two memory entries in a graph.
It defines connections between memory entries, allowing for knowledge graph construction.
*/
type Relationship struct {
	/* FromID is the ID of the source memory */
	FromID string
	/* ToID is the ID of the target memory */
	ToID string
	/* Type identifies the kind of relationship */
	Type string
	/* Metadata contains additional information about the relationship */
	Metadata map[string]interface{}
}

/*
ToCoreRelationship converts to core.Relationship from local Relationship.
This method bridges between the memory package's types and the core package's relationship types.

Returns:
  - The converted core.Relationship
*/
func (r *Relationship) ToCoreRelationship() core.Relationship {
	return core.Relationship{
		FromID:   r.FromID,
		ToID:     r.ToID,
		Type:     r.Type,
		Metadata: r.Metadata,
	}
}

/*
FromCoreRelationship converts from core.Relationship to local Relationship.
This function bridges between the core package's relationship types and the memory package's types.

Parameters:
  - rel: The core package's relationship to convert

Returns:
  - The converted local Relationship
*/
func FromCoreRelationship(rel core.Relationship) Relationship {
	return Relationship{
		FromID:   rel.FromID,
		ToID:     rel.ToID,
		Type:     rel.Type,
		Metadata: rel.Metadata,
	}
}

/*
VectorStoreProvider interface defines operations for vector-based memory storage.
This interface abstracts the vector database operations for storing and retrieving
embeddings with their associated metadata.
*/
type VectorStoreProvider interface {
	/*
		StoreVector stores a vector with the given ID and payload.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier for the vector
		  - vector: The embedding vector to store
		  - payload: Associated metadata to store with the vector

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	StoreVector(ctx context.Context, id string, vector []float32, payload map[string]interface{}) error

	/*
		Search searches for similar vectors.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - vector: The query vector to find similar vectors for
		  - limit: The maximum number of results to return
		  - filters: Optional filters to apply to the search

		Returns:
		  - A slice of SearchResult objects containing the matches
		  - An error if the operation fails, or nil on success
	*/
	Search(ctx context.Context, vector []float32, limit int, filters map[string]interface{}) ([]SearchResult, error)

	/*
		Get retrieves a specific vector by ID.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier of the vector to retrieve

		Returns:
		  - The SearchResult containing the vector and its metadata
		  - An error if the operation fails, or nil on success
	*/
	Get(ctx context.Context, id string) (*SearchResult, error)

	/*
		Delete removes a vector from the store.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier of the vector to delete

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	Delete(ctx context.Context, id string) error
}

/*
GraphStore interface defines operations for graph-based memory storage.
This interface abstracts the graph database operations for storing and
retrieving nodes, relationships, and executing graph queries.
*/
type GraphStore interface {
	/*
		CreateNode creates a new node in the graph.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier for the node
		  - labels: Classification labels for the node
		  - properties: Key-value properties to store with the node

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	CreateNode(ctx context.Context, id string, labels []string, properties map[string]interface{}) error

	/*
		CreateRelationship creates a relationship between two nodes.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - fromID: The source node ID
		  - toID: The target node ID
		  - relType: The type of relationship to create
		  - properties: Key-value properties to store with the relationship

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	CreateRelationship(ctx context.Context, fromID, toID, relType string, properties map[string]interface{}) error

	/*
		Query executes a cypher query against the graph.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - query: The Cypher query string to execute
		  - params: Parameters for the query

		Returns:
		  - Results from the query as a slice of maps
		  - An error if the operation fails, or nil on success
	*/
	Query(ctx context.Context, query string, params map[string]interface{}) ([]map[string]interface{}, error)

	/*
		DeleteNode removes a node from the graph.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier of the node to delete

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	DeleteNode(ctx context.Context, id string) error
}

/*
EmbeddingProvider interface for text-to-vector embedding services.
This interface abstracts the embedding generation service that converts
text content into vector representations for semantic operations.
*/
type EmbeddingProvider interface {
	/*
		GetEmbedding converts text to vector embeddings.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - text: The text to create an embedding for

		Returns:
		  - The vector embedding representation
		  - An error if the operation fails, or nil on success
	*/
	GetEmbedding(ctx context.Context, text string) ([]float32, error)
}

/*
SearchResult represents a single result from a vector search.
It contains the match details, including the similarity score and metadata.
*/
type SearchResult struct {
	/* ID is the unique identifier of the matched item */
	ID string
	/* Score is the similarity score of the match */
	Score float32
	/* Vector is the embedding vector of the matched item */
	Vector []float32
	/* Metadata contains additional information about the matched item */
	Metadata map[string]interface{}
}

/*
UnifiedMemoryOptions defines configuration options for unified memory.
It specifies the enabled features, connection parameters, and tuning settings
for the unified memory system.
*/
type UnifiedMemoryOptions struct {
	/* EnableVectorStore enables/disables vector-based memory features */
	EnableVectorStore bool
	/* EnableGraphStore enables/disables graph-based memory features */
	EnableGraphStore bool

	/* ExtractionThreshold sets the minimum relevance score for memory extraction */
	ExtractionThreshold float32
	/* MaxMemoriesPerQuery limits the number of memories retrieved per query */
	MaxMemoriesPerQuery int
	/* ContextTemplate is the format template for combining memories with queries */
	ContextTemplate string

	/* VectorStoreURL is the connection URL for the vector database */
	VectorStoreURL string
	/* VectorStoreAPIKey is the authentication key for the vector database */
	VectorStoreAPIKey string
	/* VectorDBCollection is the collection/index name in the vector database */
	VectorDBCollection string
	/* VectorDBDimensions is the dimensionality of the embedding vectors */
	VectorDBDimensions int
	/* VectorDBMetricType is the similarity metric to use (e.g., cosine) */
	VectorDBMetricType string

	/* GraphStoreURL is the connection URL for the graph database */
	GraphStoreURL string
	/* GraphStoreUsername is the authentication username for the graph database */
	GraphStoreUsername string
	/* GraphStorePassword is the authentication password for the graph database */
	GraphStorePassword string
	/* GraphStoreDatabase is the database name in the graph database */
	GraphStoreDatabase string
}

/*
DefaultUnifiedMemoryOptions returns default options.
It provides sensible defaults for the unified memory configuration.

Returns:
  - A pointer to an initialized UnifiedMemoryOptions struct with default values
*/
func DefaultUnifiedMemoryOptions() *UnifiedMemoryOptions {
	return &UnifiedMemoryOptions{
		EnableVectorStore:   true,
		EnableGraphStore:    false,
		ExtractionThreshold: 0.7,
		MaxMemoriesPerQuery: 5,
		VectorDBDimensions:  1536, // Default for OpenAI embeddings
		VectorDBMetricType:  "cosine",
		ContextTemplate:     "Previous relevant memories:\n{{.Memories}}\n\nCurrent query: {{.Query}}",
	}
}

/*
UnifiedMemory implements an enhanced memory system with both vector and graph capabilities.
It provides a comprehensive memory solution that combines traditional key-value storage
with vector-based semantic search and graph-based relationship tracking.
*/
type UnifiedMemory struct {
	/* baseStore is the underlying simple key-value memory store */
	baseStore core.Memory
	/* vectorStore provides vector-based semantic search capabilities */
	vectorStore VectorStoreProvider
	/* graphStore provides graph-based relationship tracking */
	graphStore GraphStore
	/* embeddingProvider generates vector embeddings from text */
	embeddingProvider EmbeddingProvider
	/* options contains the configuration settings */
	options *UnifiedMemoryOptions
	/* mutex protects concurrent access to memory structures */
	mutex sync.RWMutex

	/* memoryData is an in-memory cache of memory entries */
	memoryData map[string]*EnhancedMemoryEntry
	/* contextTemplate is the compiled template for context generation */
	contextTemplate *template.Template
}

/*
NewUnifiedMemory creates a new unified memory system.
It initializes the memory system with the provided stores and configuration.

Parameters:
  - baseStore: The underlying core Memory implementation
  - embeddingProvider: The service that generates embeddings from text
  - options: Configuration options for the unified memory

Returns:
  - A pointer to the initialized UnifiedMemory
  - An error if initialization fails, or nil on success
*/
func NewUnifiedMemory(baseStore core.Memory, embeddingProvider EmbeddingProvider, options *UnifiedMemoryOptions) (*UnifiedMemory, error) {
	if options == nil {
		options = DefaultUnifiedMemoryOptions()
	}

	var vectorStore VectorStoreProvider
	var graphStore GraphStore
	var err error

	// Initialize vector store if enabled
	if options.EnableVectorStore {
		vectorStore, err = NewQDrantStore(options.VectorStoreURL, options.VectorStoreAPIKey, options.VectorDBCollection, options.VectorDBDimensions)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize vector store: %w", err)
		}
	}

	// Initialize graph store if enabled
	if options.EnableGraphStore {
		graphStore, err = NewNeo4jStore(options.GraphStoreURL, options.GraphStoreUsername, options.GraphStorePassword, options.GraphStoreDatabase)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize graph store: %w", err)
		}
	}

	// Parse context template
	tmpl, err := template.New("context").Parse(options.ContextTemplate)
	if err != nil {
		return nil, fmt.Errorf("failed to parse context template: %w", err)
	}

	return &UnifiedMemory{
		baseStore:         baseStore,
		vectorStore:       vectorStore,
		graphStore:        graphStore,
		embeddingProvider: embeddingProvider,
		options:           options,
		memoryData:        make(map[string]*EnhancedMemoryEntry),
		contextTemplate:   tmpl,
	}, nil
}

/*
Store implements the core Memory interface.
It stores a key-value pair in the base memory store.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - key: The unique identifier for the value
  - value: The data to store

Returns:
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) Store(ctx context.Context, key string, value interface{}) error {
	// Use the base memory store for regular key-value storage
	return um.baseStore.Store(ctx, key, value)
}

/*
Retrieve implements the core Memory interface.
It retrieves a value from the base memory store by key.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - key: The key to look up

Returns:
  - The stored value if found
  - An error if the key doesn't exist or retrieval fails
*/
func (um *UnifiedMemory) Retrieve(ctx context.Context, key string) (interface{}, error) {
	// Use the base memory store for regular key-value retrieval
	return um.baseStore.Retrieve(ctx, key)
}

/*
Search implements the core Memory interface.
It performs a search across both the base memory store and the vector store
if enabled, combining and ranking the results.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - query: The search query text
  - limit: The maximum number of results to return (0 means no limit)

Returns:
  - A slice of MemoryEntry objects containing the search results
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) Search(ctx context.Context, query string, limit int) ([]core.MemoryEntry, error) {
	// First, search in the base memory store
	baseResults, err := um.baseStore.Search(ctx, query, limit)
	if err != nil {
		return nil, err
	}

	// If vector store is enabled, also search there
	if um.options.EnableVectorStore && um.vectorStore != nil {
		// Get embedding for query
		embedding, err := um.embeddingProvider.GetEmbedding(ctx, query)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to get embedding for query: %v", err))
		} else {
			// Search the vector store
			vectorResults, err := um.vectorStore.Search(ctx, embedding, limit, nil)
			if err != nil {
				errnie.Info(fmt.Sprintf("Failed to search vector store: %v", err))
			} else {
				// Convert vector results to core.MemoryEntry
				for _, result := range vectorResults {
					content, ok := result.Metadata["content"].(string)
					if !ok {
						continue
					}
					baseResults = append(baseResults, core.MemoryEntry{
						Key:   result.ID,
						Value: content,
						Score: float64(result.Score),
					})
				}
			}
		}
	}

	// Deduplicate and limit results
	seen := make(map[string]bool)
	uniqueResults := make([]core.MemoryEntry, 0, len(baseResults))

	for _, result := range baseResults {
		if !seen[result.Key] {
			seen[result.Key] = true
			uniqueResults = append(uniqueResults, result)
		}
	}

	// Sort by score
	sort.Slice(uniqueResults, func(i, j int) bool {
		return uniqueResults[i].Score > uniqueResults[j].Score
	})

	// Apply limit
	if limit > 0 && len(uniqueResults) > limit {
		uniqueResults = uniqueResults[:limit]
	}

	return uniqueResults, nil
}

/*
Clear implements the core Memory interface.
It clears the base memory store and the in-memory data structure,
but preserves the vector and graph stores.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation

Returns:
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) Clear(ctx context.Context) error {
	// Clear the base store
	err := um.baseStore.Clear(ctx)
	if err != nil {
		return err
	}

	// Clear in-memory data
	um.mutex.Lock()
	um.memoryData = make(map[string]*EnhancedMemoryEntry)
	um.mutex.Unlock()

	// We don't clear the vector or graph stores by default
	// as they might contain shared data across multiple agents

	return nil
}

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
RetrieveMemoriesByVector searches for memories using vector similarity.
It searches the vector store for memories similar to the provided query.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - query: The search query text
  - agentID: The agent ID to filter by (empty string for no filter)
  - limit: The maximum number of results to return
  - threshold: The minimum similarity score to include in results

Returns:
  - A slice of EnhancedMemoryEntry objects matching the query
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) RetrieveMemoriesByVector(ctx context.Context, query string, agentID string, limit int, threshold float32) ([]EnhancedMemoryEntry, error) {
	if !um.options.EnableVectorStore || um.vectorStore == nil {
		return nil, errors.New("vector store not enabled or not available")
	}

	// Get embedding for query
	embedding, err := um.embeddingProvider.GetEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding for query: %w", err)
	}

	// Prepare filters
	filters := make(map[string]interface{})
	if agentID != "" {
		filters["agent_id"] = agentID
	}

	// Search the vector store
	results, err := um.vectorStore.Search(ctx, embedding, limit, filters)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector store: %w", err)
	}

	// Convert results to EnhancedMemoryEntry and apply threshold
	memories := make([]EnhancedMemoryEntry, 0, len(results))
	for _, result := range results {
		if result.Score < threshold {
			continue
		}

		// Extract fields from metadata
		content, _ := result.Metadata["content"].(string)
		agentID, _ := result.Metadata["agent_id"].(string)
		source, _ := result.Metadata["source"].(string)
		memType, _ := result.Metadata["type"].(string)
		createdAt, _ := result.Metadata["created_at"].(time.Time)

		// Create memory entry
		entry := EnhancedMemoryEntry{
			ID:          result.ID,
			AgentID:     agentID,
			Content:     content,
			Embedding:   result.Vector,
			Type:        MemoryType(memType),
			Source:      source,
			CreatedAt:   createdAt,
			AccessCount: 1,
			LastAccess:  time.Now(),
			Metadata:    result.Metadata,
		}

		memories = append(memories, entry)
	}

	return memories, nil
}

/*
RetrieveMemoriesByGraph searches for memories using graph relationships.
It executes a Cypher query against the graph store to find related memories.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - cypherQuery: The Cypher query to execute against the graph database
  - params: Parameters for the Cypher query

Returns:
  - A slice of EnhancedMemoryEntry objects matching the query
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) RetrieveMemoriesByGraph(ctx context.Context, cypherQuery string, params map[string]interface{}) ([]EnhancedMemoryEntry, error) {
	if !um.options.EnableGraphStore || um.graphStore == nil {
		return nil, errors.New("graph store not enabled or not available")
	}

	// Execute query
	results, err := um.graphStore.Query(ctx, cypherQuery, params)
	if err != nil {
		return nil, fmt.Errorf("failed to execute graph query: %w", err)
	}

	// Convert results to EnhancedMemoryEntry
	memories := make([]EnhancedMemoryEntry, 0, len(results))
	for _, result := range results {
		// Extract memory node properties
		node, ok := result["memory"].(map[string]interface{})
		if !ok {
			continue
		}

		id, _ := node["id"].(string)
		content, _ := node["content"].(string)
		agentID, _ := node["agent_id"].(string)
		source, _ := node["source"].(string)
		memType, _ := node["type"].(string)
		createdAtStr, _ := node["created_at"].(string)

		// Parse created at
		var createdAt time.Time
		if createdAtStr != "" {
			createdAt, _ = time.Parse(time.RFC3339, createdAtStr)
		}

		// Create memory entry
		entry := EnhancedMemoryEntry{
			ID:          id,
			AgentID:     agentID,
			Content:     content,
			Type:        MemoryType(memType),
			Source:      source,
			CreatedAt:   createdAt,
			AccessCount: 1,
			LastAccess:  time.Now(),
			Metadata:    node,
		}

		memories = append(memories, entry)
	}

	return memories, nil
}

/*
CreateRelationship creates a relationship between two memories.
It establishes a typed connection between memory nodes in the graph store.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - relationship: The relationship to create

Returns:
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) CreateRelationship(ctx context.Context, relationship Relationship) error {
	if !um.options.EnableGraphStore || um.graphStore == nil {
		return errors.New("graph store not enabled or not available")
	}

	// Create relationship in the graph store
	return um.graphStore.CreateRelationship(
		ctx,
		relationship.FromID,
		relationship.ToID,
		relationship.Type,
		relationship.Metadata,
	)
}

/*
GetRelatedMemories retrieves memories related to a specific memory.
It finds connected memories in the graph store based on relationship types.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - memoryID: The ID of the memory to find relations for
  - relationshipType: The type of relationship to follow (empty for any)
  - maxDepth: The maximum traversal depth in the graph

Returns:
  - A slice of EnhancedMemoryEntry objects related to the specified memory
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) GetRelatedMemories(ctx context.Context, memoryID string, relationshipType string, maxDepth int) ([]EnhancedMemoryEntry, error) {
	if !um.options.EnableGraphStore || um.graphStore == nil {
		return nil, errors.New("graph store not enabled or not available")
	}

	// Default max depth
	if maxDepth <= 0 {
		maxDepth = 1
	}

	// Build Cypher query to find related memories
	var query string
	params := map[string]interface{}{
		"memoryID": memoryID,
	}

	if relationshipType != "" {
		query = fmt.Sprintf(
			`MATCH (m:Memory {id: $memoryID})-[r:%s*1..%d]-(related:Memory)
			 RETURN related as memory`,
			relationshipType, maxDepth,
		)
	} else {
		query = fmt.Sprintf(
			`MATCH (m:Memory {id: $memoryID})-[r*1..%d]-(related:Memory)
			 RETURN related as memory`,
			maxDepth,
		)
	}

	// Execute query and convert results
	return um.RetrieveMemoriesByGraph(ctx, query, params)
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
	// This is a simplified implementation
	// In a real implementation, we'd use the LLM to identify important information

	// Split text into paragraphs
	paragraphs := strings.Split(text, "\n\n")
	memories := make([]string, 0)

	for _, paragraph := range paragraphs {
		paragraph = strings.TrimSpace(paragraph)
		if len(paragraph) < 20 {
			continue // Skip short paragraphs
		}

		// Here we would normally check if the paragraph is important/interesting enough
		// For now, just store paragraphs longer than a certain length
		if len(paragraph) > 100 {
			memories = append(memories, paragraph)
		}
	}

	// Store the extracted memories
	for _, memory := range memories {
		_, err := um.StoreMemory(ctx, agentID, memory, MemoryTypePersonal, source, nil)
		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to store extracted memory: %v", err))
		}
	}

	return memories, nil
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

	// Retrieve relevant memories
	memories, err := um.RetrieveMemoriesByVector(
		ctx,
		query,
		agentID,
		um.options.MaxMemoriesPerQuery,
		um.options.ExtractionThreshold,
	)
	if err != nil {
		errnie.Info(fmt.Sprintf("Failed to retrieve relevant memories: %v", err))
		return query, nil
	}

	// If no memories, just return the query
	if len(memories) == 0 {
		return query, nil
	}

	// Format memories as text
	var memoriesText strings.Builder
	for i, memory := range memories {
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

/*
QDrantStore implements vector storage using QDrant.
It provides a vector database implementation for storing and
retrieving vector embeddings with the QDrant vector database.
*/
type QDrantStore struct {
	/* url is the connection URL for the QDrant server */
	url string
	/* apiKey is the authentication key for the QDrant API */
	apiKey string
	/* collection is the name of the collection in QDrant */
	collection string
	/* dimensions is the size of vectors stored in this collection */
	dimensions int
}

/*
NewQDrantStore creates a new QDrant store.
It initializes a connection to a QDrant vector database.

Parameters:
  - url: The URL of the QDrant server
  - apiKey: The authentication key for the QDrant API
  - collection: The name of the collection to use
  - dimensions: The dimensionality of vectors to store

Returns:
  - A pointer to the initialized QDrantStore
  - An error if initialization fails, or nil on success
*/
func NewQDrantStore(url, apiKey, collection string, dimensions int) (*QDrantStore, error) {
	// Placeholder implementation
	// In a real implementation, we would validate the connection to QDrant

	return &QDrantStore{
		url:        url,
		apiKey:     apiKey,
		collection: collection,
		dimensions: dimensions,
	}, nil
}

/*
StoreVector stores a vector with the given ID and payload.
It adds a vector embedding and its metadata to the QDrant collection.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - id: The unique identifier for the vector
  - vector: The embedding vector to store
  - payload: Associated metadata to store with the vector

Returns:
  - An error if the operation fails, or nil on success
*/
func (q *QDrantStore) StoreVector(ctx context.Context, id string, vector []float32, payload map[string]interface{}) error {
	// Placeholder implementation
	errnie.Info(fmt.Sprintf("QDrantStore: Storing vector for ID %s with %d dimensions", id, len(vector)))
	return nil
}

/*
Search searches for similar vectors.
It finds vectors in the QDrant collection similar to the query vector.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - vector: The query vector to find similar vectors for
  - limit: The maximum number of results to return
  - filters: Optional filters to apply to the search

Returns:
  - A slice of SearchResult objects containing the matches
  - An error if the operation fails, or nil on success
*/
func (q *QDrantStore) Search(ctx context.Context, vector []float32, limit int, filters map[string]interface{}) ([]SearchResult, error) {
	// Placeholder implementation
	errnie.Info(fmt.Sprintf("QDrantStore: Searching for similar vectors to %d dimensions", len(vector)))
	return []SearchResult{}, nil
}

/*
Get retrieves a specific vector by ID.
It looks up a vector in the QDrant collection by its unique identifier.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - id: The unique identifier of the vector to retrieve

Returns:
  - The SearchResult containing the vector and its metadata
  - An error if the operation fails, or nil on success
*/
func (q *QDrantStore) Get(ctx context.Context, id string) (*SearchResult, error) {
	// Placeholder implementation
	errnie.Info(fmt.Sprintf("QDrantStore: Getting vector for ID %s", id))
	return nil, nil
}

/*
Delete removes a vector from the store.
It removes a vector from the QDrant collection by its unique identifier.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - id: The unique identifier of the vector to delete

Returns:
  - An error if the operation fails, or nil on success
*/
func (q *QDrantStore) Delete(ctx context.Context, id string) error {
	// Placeholder implementation
	errnie.Info(fmt.Sprintf("QDrantStore: Deleting vector for ID %s", id))
	return nil
}

/*
Neo4jStore implements graph storage using Neo4j.
It provides a graph database implementation for storing and
querying nodes and relationships with the Neo4j graph database.
*/
type Neo4jStore struct {
	/* url is the connection URL for the Neo4j server */
	url string
	/* username is the authentication username for Neo4j */
	username string
	/* password is the authentication password for Neo4j */
	password string
	/* database is the name of the database in Neo4j */
	database string
}

/*
NewNeo4jStore creates a new Neo4j store.
It initializes a connection to a Neo4j graph database.

Parameters:
  - url: The URL of the Neo4j server
  - username: The authentication username for Neo4j
  - password: The authentication password for Neo4j
  - database: The name of the database to use

Returns:
  - A pointer to the initialized Neo4jStore
  - An error if initialization fails, or nil on success
*/
func NewNeo4jStore(url, username, password, database string) (*Neo4jStore, error) {
	// Placeholder implementation
	// In a real implementation, we would validate the connection to Neo4j

	return &Neo4jStore{
		url:      url,
		username: username,
		password: password,
		database: database,
	}, nil
}

/*
CreateNode creates a new node in the graph.
It adds a node with the specified labels and properties to the Neo4j graph.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - id: The unique identifier for the node
  - labels: Classification labels for the node
  - properties: Key-value properties to store with the node

Returns:
  - An error if the operation fails, or nil on success
*/
func (n *Neo4jStore) CreateNode(ctx context.Context, id string, labels []string, properties map[string]interface{}) error {
	// Placeholder implementation
	errnie.Info(fmt.Sprintf("Neo4jStore: Creating node with ID %s and labels %v", id, labels))
	return nil
}

/*
CreateRelationship creates a relationship between two nodes.
It establishes a typed connection between nodes in the Neo4j graph.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - fromID: The source node ID
  - toID: The target node ID
  - relType: The type of relationship to create
  - properties: Key-value properties to store with the relationship

Returns:
  - An error if the operation fails, or nil on success
*/
func (n *Neo4jStore) CreateRelationship(ctx context.Context, fromID, toID, relType string, properties map[string]interface{}) error {
	// Placeholder implementation
	errnie.Info(fmt.Sprintf("Neo4jStore: Creating relationship %s from %s to %s", relType, fromID, toID))
	return nil
}

/*
Query executes a cypher query against the graph.
It runs a Cypher language query against the Neo4j database.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - query: The Cypher query string to execute
  - params: Parameters for the query

Returns:
  - Results from the query as a slice of maps
  - An error if the operation fails, or nil on success
*/
func (n *Neo4jStore) Query(ctx context.Context, query string, params map[string]interface{}) ([]map[string]interface{}, error) {
	// Placeholder implementation
	errnie.Info(fmt.Sprintf("Neo4jStore: Executing query: %s", query))
	return []map[string]interface{}{}, nil
}

/*
DeleteNode removes a node from the graph.
It deletes a node from the Neo4j graph by its unique identifier.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - id: The unique identifier of the node to delete

Returns:
  - An error if the operation fails, or nil on success
*/
func (n *Neo4jStore) DeleteNode(ctx context.Context, id string) error {
	// Placeholder implementation
	errnie.Info(fmt.Sprintf("Neo4jStore: Deleting node with ID %s", id))
	return nil
}

/*
OpenAIEmbeddingProvider provides embeddings using OpenAI.
It implements the EmbeddingProvider interface using OpenAI's
embedding models to convert text to vector representations.
*/
type OpenAIEmbeddingProvider struct {
	/* apiKey is the authentication key for the OpenAI API */
	apiKey string
	/* model is the specific embedding model to use */
	model string
}

/*
NewOpenAIEmbeddingProvider creates a new OpenAI embedding provider.
It initializes a provider that uses OpenAI's API to generate embeddings.

Parameters:
  - apiKey: The authentication key for the OpenAI API
  - model: The name of the embedding model to use

Returns:
  - A pointer to the initialized OpenAIEmbeddingProvider
*/
func NewOpenAIEmbeddingProvider(apiKey, model string) *OpenAIEmbeddingProvider {
	if model == "" {
		model = "text-embedding-ada-002"
	}

	return &OpenAIEmbeddingProvider{
		apiKey: apiKey,
		model:  model,
	}
}

/*
GetEmbedding converts text to vector embeddings.
It transforms text into a numerical vector representation using OpenAI's models.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - text: The text to create an embedding for

Returns:
  - The vector embedding representation
  - An error if the operation fails, or nil on success
*/
func (o *OpenAIEmbeddingProvider) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	// Placeholder implementation
	// In a real implementation, this would call the OpenAI API
	errnie.Info(fmt.Sprintf("OpenAIEmbeddingProvider: Getting embedding for text of length %d", len(text)))

	// Return a dummy embedding of the right size
	return make([]float32, 1536), nil
}
