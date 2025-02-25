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
	"strings"
	"sync"
	"text/template"
	"time"

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
		VectorDBDimensions:  3072, // Default for OpenAI embeddings
		VectorDBMetricType:  "cosine",
		ContextTemplate:     "Previous relevant memories:\n{{.Memories}}\n\nCurrent query: {{.Query}}",
		// Default connection values will be added by ConnectionAwareMemoryOptions
	}
}

/*
ConnectionAwareMemoryOptions extends DefaultUnifiedMemoryOptions with environment-aware connection settings.
It attempts to automatically discover and configure the correct connection parameters for memory stores.

Parameters:
  - baseOptions: Optional existing options to extend. If nil, DefaultUnifiedMemoryOptions() will be used.

Returns:
  - A pointer to an UnifiedMemoryOptions struct with connection-aware values
*/
func ConnectionAwareMemoryOptions(baseOptions *UnifiedMemoryOptions) *UnifiedMemoryOptions {
	if baseOptions == nil {
		baseOptions = DefaultUnifiedMemoryOptions()
	}

	// Set sensible defaults for vector store (Qdrant)
	if baseOptions.EnableVectorStore {
		// Check environment variables first
		if url := os.Getenv("QDRANT_URL"); url != "" {
			baseOptions.VectorStoreURL = url
		} else {
			baseOptions.VectorStoreURL = "http://localhost:6333" // Explicitly use the REST API port
		}

		if apiKey := os.Getenv("QDRANT_API_KEY"); apiKey != "" {
			baseOptions.VectorStoreAPIKey = apiKey
		} else {
			baseOptions.VectorStoreAPIKey = "gKzti5QyA5KeLQYQFLA1T6pT3GYE9pza" // Default dev API key
		}

		if collection := os.Getenv("QDRANT_COLLECTION"); collection != "" {
			baseOptions.VectorDBCollection = collection
		} else if baseOptions.VectorDBCollection == "" {
			baseOptions.VectorDBCollection = "long-term-memory" // Default collection name
		}
	}

	// Set sensible defaults for graph store (Neo4j)
	if baseOptions.EnableGraphStore {
		// Check environment variables first
		if url := os.Getenv("NEO4J_URL"); url != "" {
			baseOptions.GraphStoreURL = url
		} else {
			baseOptions.GraphStoreURL = "bolt://localhost:7687" // Default Neo4j URL
		}

		if username := os.Getenv("NEO4J_USERNAME"); username != "" {
			baseOptions.GraphStoreUsername = username
		} else {
			baseOptions.GraphStoreUsername = "neo4j" // Default Neo4j username
		}

		if password := os.Getenv("NEO4J_PASSWORD"); password != "" {
			baseOptions.GraphStorePassword = password
		} else {
			baseOptions.GraphStorePassword = "securepassword" // Default Neo4j password
		}

		if database := os.Getenv("NEO4J_DATABASE"); database != "" {
			baseOptions.GraphStoreDatabase = database
		} else if baseOptions.GraphStoreDatabase == "" {
			baseOptions.GraphStoreDatabase = "neo4j" // Default database name
		}
	}

	return baseOptions
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
		options = ConnectionAwareMemoryOptions(nil)
	} else {
		// Apply connection-aware defaults to any unset connection parameters
		options = ConnectionAwareMemoryOptions(options)
	}

	var vectorStore VectorStoreProvider
	var graphStore GraphStore
	var err error
	var warnings []string

	// Initialize vector store if enabled
	if options.EnableVectorStore {
		vectorStore, err = NewQDrantStore(options.VectorStoreURL, options.VectorStoreAPIKey, options.VectorDBCollection, options.VectorDBDimensions)
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("Vector store initialization failed: %v", err))
			fmt.Printf("Warning: Vector store initialization failed: %v\n", err)
			fmt.Println("Memory will continue with reduced functionality (without vector search).")
			options.EnableVectorStore = false
		}
	}

	// Initialize graph store if enabled
	if options.EnableGraphStore {
		graphStore, err = NewNeo4jStore(options.GraphStoreURL, options.GraphStoreUsername, options.GraphStorePassword, options.GraphStoreDatabase)
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("Graph store initialization failed: %v", err))
			fmt.Printf("Warning: Graph store initialization failed: %v\n", err)
			fmt.Println("Memory will continue with reduced functionality (without graph relationships).")
			options.EnableGraphStore = false

			// Try alternative Neo4j connection formats if this was likely a connection issue
			if strings.Contains(err.Error(), "connect") || strings.Contains(err.Error(), "dial") {
				fmt.Println("Attempting alternative Neo4j connection formats...")

				// Try with neo4j:// protocol
				altURL := strings.Replace(options.GraphStoreURL, "bolt://", "neo4j://", 1)
				if altURL != options.GraphStoreURL {
					fmt.Printf("Trying neo4j:// protocol: %s\n", altURL)
					graphStore, err = NewNeo4jStore(altURL, options.GraphStoreUsername, options.GraphStorePassword, options.GraphStoreDatabase)
					if err == nil {
						fmt.Println("Success! Connected with neo4j:// protocol.")
						options.EnableGraphStore = true
						options.GraphStoreURL = altURL
					}
				}

				// If still not connected, try with http:// protocol for browser connection
				if err != nil && strings.HasPrefix(options.GraphStoreURL, "bolt://") {
					httpURL := strings.Replace(options.GraphStoreURL, "bolt://", "http://", 1)
					httpURL = strings.Replace(httpURL, "7687", "7474", 1)
					fmt.Printf("Trying http:// protocol: %s\n", httpURL)
					graphStore, err = NewNeo4jStore(httpURL, options.GraphStoreUsername, options.GraphStorePassword, options.GraphStoreDatabase)
					if err == nil {
						fmt.Println("Success! Connected with http:// protocol.")
						options.EnableGraphStore = true
						options.GraphStoreURL = httpURL
					}
				}
			}
		}
	}

	// Parse context template
	tmpl, err := template.New("context").Parse(options.ContextTemplate)
	if err != nil {
		return nil, fmt.Errorf("failed to parse context template: %w", err)
	}

	// Print status of memory features
	if options.EnableVectorStore {
		fmt.Println("Vector store (Qdrant) successfully connected.")
	}
	if options.EnableGraphStore {
		fmt.Println("Graph store (Neo4j) successfully connected.")
	}
	if !options.EnableVectorStore && !options.EnableGraphStore {
		fmt.Println("Warning: Running with basic memory only (no vector or graph features).")
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
