/*
Package memory provides various memory storage implementations for Caramba agents.
This package offers different memory storage solutions, from simple in-memory
stores to sophisticated vector and graph-based memory implementations that
support semantic search and complex relationships between memory items.
*/
package memory

import (
	"context"
	"fmt"
	"os"
	"sort"
	"sync"
	"text/template"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/hub"
)

/*
MemoryType represents the type of memory (personal or global).
It distinguishes between agent-specific memories and shared global memories.
*/
type MemoryType string

const (
	// MemoryTypePersonal represents an agent's personal memory
	MemoryTypePersonal MemoryType = "personal"

	// MemoryTypeGlobal represents shared global memory
	MemoryTypeGlobal MemoryType = "global"
)

/*
MemoryStoreType represents the backend memory store type.
It identifies the storage implementation used for memories.
*/
type MemoryStoreType string

const (
	// MemoryStoreVector represents vector-based memory storage (e.g., QDrant)
	MemoryStoreVector MemoryStoreType = "vector"

	// MemoryStoreGraph represents graph-based memory storage (e.g., Neo4j)
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
	Metadata map[string]string
	/* Score is the relevance/importance score for this memory */
	Score float32
	/* ImportanceScore is the algorithmically determined importance */
	ImportanceScore float32
	/* RelevanceCache caches relevance scores to common queries */
	RelevanceCache *RelevanceCache
	/* UsageStats tracks detailed usage statistics for the memory */
	UsageStats *MemoryUsageStats
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
	Metadata map[string]string
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
		EnableGraphStore:    true,
		ExtractionThreshold: 0.7,
		MaxMemoriesPerQuery: 5,
		VectorDBDimensions:  3072, // Default for OpenAI embeddings
		VectorDBMetricType:  "cosine",
		ContextTemplate:     "Previous relevant memories:\n{{.Memories}}\n\nCurrent query: {{.Query}}",
		VectorDBCollection:  "long-term-memory", // Default collection name
		// Default connection values will be added by ConnectionAwareMemoryOptions
	}
}

/*
UnifiedMemory implements an enhanced memory system with both vector and graph capabilities.
It provides a comprehensive memory solution that combines traditional key-value storage
with vector-based semantic search and graph-based relationship tracking.
*/
type UnifiedMemory struct {
	hub *hub.Queue
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
func NewUnifiedMemory(baseStore core.Memory, options *UnifiedMemoryOptions) *UnifiedMemory {
	var vectorStore VectorStoreProvider
	var graphStore GraphStore
	var err error

	q := hub.NewQueue()

	// If options is nil, use default options
	if options == nil {
		options = DefaultUnifiedMemoryOptions()
	}

	if options.VectorStoreAPIKey == "" {
		options.VectorStoreAPIKey = os.Getenv("QDRANT_API_KEY")
	}

	// Create a background context for initialization
	ctx := context.Background()

	// Initialize vector store if enabled
	if options.EnableVectorStore {
		vectorStore, err = NewQDrantStore(
			options.VectorStoreURL,
			options.VectorStoreAPIKey,
			options.VectorDBCollection,
			options.VectorDBDimensions,
		)
		if err != nil {
			fmt.Printf("Warning: Vector store initialization failed: %v\n", err)
			fmt.Println("Memory will continue with reduced functionality (without vector search).")
			options.EnableVectorStore = false
		}
	}

	// Initialize graph store if enabled
	if options.EnableGraphStore {
		graphStore, err = NewNeo4jStore(ctx, "bolt://localhost:7687", "neo4j", "securepassword", "neo4j")

		if err != nil {
			q.Add(hub.NewEvent(
				"neo4j",
				"error",
				"connection",
				hub.EventTypeError,
				err.Error(),
				map[string]string{},
			))
		}
	}

	// Parse context template
	tmpl, err := template.New("context").Parse(options.ContextTemplate)
	if err != nil {
		q.Add(hub.NewEvent(
			"template",
			"ui",
			"error",
			hub.EventTypeError,
			err.Error(),
			map[string]string{},
		))
	}

	// Print status of memory features
	if options.EnableVectorStore {
		q.Add(hub.NewEvent(
			"vector",
			"ui",
			"success",
			hub.EventTypeStatus,
			"Vector store (Qdrant) successfully connected.",
			map[string]string{},
		))
	}
	if options.EnableGraphStore {
		q.Add(hub.NewEvent(
			"graph",
			"ui",
			"success",
			hub.EventTypeStatus,
			"Graph store (Neo4j) successfully connected.",
			map[string]string{},
		))
	}
	if !options.EnableVectorStore && !options.EnableGraphStore {
		q.Add(hub.NewEvent(
			"memory",
			"ui",
			"warning",
			hub.EventTypeWarning,
			"Running with basic memory only (no vector or graph features).",
			map[string]string{},
		))
	}

	memory := &UnifiedMemory{
		hub:               q,
		baseStore:         baseStore,
		vectorStore:       vectorStore,
		graphStore:        graphStore,
		embeddingProvider: NewOpenAIEmbeddingProvider(os.Getenv("OPENAI_API_KEY"), "text-embedding-3-large"),
		options:           options,
		memoryData:        make(map[string]*EnhancedMemoryEntry),
		contextTemplate:   tmpl,
	}

	return memory
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
			um.hub.Add(hub.NewEvent(
				"vector",
				"ui",
				"error",
				hub.EventTypeError,
				err.Error(),
				map[string]string{},
			))
		} else {
			// Search the vector store
			vectorResults, err := um.vectorStore.Search(ctx, embedding, limit, nil)
			if err != nil {
				um.hub.Add(hub.NewEvent(
					"vector",
					"ui",
					"error",
					hub.EventTypeError,
					err.Error(),
					map[string]string{},
				))
			} else {
				// Convert vector results to core.MemoryEntry
				for _, result := range vectorResults {
					content := result.Metadata["content"]
					if content == "" {
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
ForgetMemory removes a specific memory from all storage systems
*/
func (um *UnifiedMemory) ForgetMemory(ctx context.Context, memoryID string) error {
	um.mutex.Lock()
	defer um.mutex.Unlock()

	// Remove from in-memory cache
	delete(um.memoryData, memoryID)

	// Remove from vector store if enabled
	if um.options.EnableVectorStore && um.vectorStore != nil {
		if err := um.vectorStore.Delete(ctx, memoryID); err != nil {
			um.hub.Add(hub.NewEvent(
				"vector",
				"ui",
				"error",
				hub.EventTypeError,
				err.Error(),
				map[string]string{},
			))
		}
	}

	// Remove from graph store if enabled
	if um.options.EnableGraphStore && um.graphStore != nil {
		if err := um.graphStore.DeleteNode(ctx, memoryID); err != nil {
			um.hub.Add(hub.NewEvent(
				"neo4j",
				"ui",
				"error",
				hub.EventTypeError,
				err.Error(),
				map[string]string{},
			))
		}
	}

	// The core.Memory interface doesn't have a Delete method directly
	// Memory clearing is typically done at a higher level, so we'll
	// simply remove our in-memory tracking of this entry
	// For a complete solution, we would need to implement a Delete method
	// in the specific baseStore implementation being used

	return nil
}
