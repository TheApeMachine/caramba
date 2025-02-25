package core

import (
	"time"
)

// MemoryType represents the type of memory
type MemoryType string

const (
	// MemoryTypePersonal represents an agent's personal memory
	MemoryTypePersonal MemoryType = "personal"

	// MemoryTypeGlobal represents shared global memory
	MemoryTypeGlobal MemoryType = "global"
)

// MemoryStoreType represents the backend memory store type
type MemoryStoreType string

const (
	// MemoryStoreVector represents vector-based memory storage (e.g., QDrant)
	MemoryStoreVector MemoryStoreType = "vector"

	// MemoryStoreGraph represents graph-based memory storage (e.g., Neo4j)
	MemoryStoreGraph MemoryStoreType = "graph"
)

// EnhancedMemoryEntry represents a single memory entry with extended metadata
type EnhancedMemoryEntry struct {
	ID          string                 // Unique identifier for the memory
	AgentID     string                 // ID of the agent who owns the memory (empty for global)
	Content     string                 // The actual memory content
	Embedding   []float32              // Vector embedding of the content
	Type        MemoryType             // Type of memory (personal/global)
	Source      string                 // Source of the memory (conversation, document, etc.)
	CreatedAt   time.Time              // When the memory was created
	AccessCount int                    // How many times this memory has been accessed
	LastAccess  time.Time              // When this memory was last accessed
	Metadata    map[string]interface{} // Additional metadata
}

// Relationship represents a relationship between two memory entries in a graph
type Relationship struct {
	FromID   string                 // ID of the source memory
	ToID     string                 // ID of the target memory
	Type     string                 // Type of relationship
	Metadata map[string]interface{} // Additional metadata about the relationship
}

// MemoryOptions defines configuration options for memory systems
type MemoryOptions struct {
	// Vector store options
	VectorStoreURL     string
	VectorStoreAPIKey  string
	VectorDBCollection string
	VectorDBDimensions int
	VectorDBMetricType string

	// Graph store options
	GraphStoreURL      string
	GraphStoreUsername string
	GraphStorePassword string
	GraphStoreDatabase string

	// Embedding options
	EmbeddingProvider string
	EmbeddingModel    string
	EmbeddingAPIKey   string

	// Memory extraction options
	ExtractionThreshold float32
	MaxMemoriesPerQuery int
	ContextTemplate     string
}

// DefaultMemoryOptions returns the default memory options
func DefaultMemoryOptions() *MemoryOptions {
	return &MemoryOptions{
		VectorDBDimensions:  1536, // For OpenAI embeddings
		VectorDBMetricType:  "cosine",
		ExtractionThreshold: 0.7,
		MaxMemoriesPerQuery: 5,
		ContextTemplate:     "Previous relevant memories:\n{{.Memories}}\n\nCurrent query: {{.Query}}",
	}
}

// MemoryScore represents a memory's relevance score
type MemoryScore struct {
	MemoryID string
	Score    float32
}

// MemoryScoreList is a sortable list of memory scores
type MemoryScoreList []MemoryScore

// Len returns the length of the list (for sort interface)
func (m MemoryScoreList) Len() int {
	return len(m)
}

// Less compares scores (for sort interface)
func (m MemoryScoreList) Less(i, j int) bool {
	return m[i].Score > m[j].Score // Higher scores first
}

// Swap swaps elements (for sort interface)
func (m MemoryScoreList) Swap(i, j int) {
	m[i], m[j] = m[j], m[i]
}
