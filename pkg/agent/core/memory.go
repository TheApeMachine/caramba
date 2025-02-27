package core

import (
	"context"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/errnie"
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
	ID          string            // Unique identifier for the memory
	AgentID     string            // ID of the agent who owns the memory (empty for global)
	Content     string            // The actual memory content
	Embedding   []float32         // Vector embedding of the content
	Type        MemoryType        // Type of memory (personal/global)
	Source      string            // Source of the memory (conversation, document, etc.)
	CreatedAt   time.Time         // When the memory was created
	AccessCount int               // How many times this memory has been accessed
	LastAccess  time.Time         // When this memory was last accessed
	Metadata    map[string]string // Additional metadata
}

// Relationship represents a relationship between two memory entries in a graph
type Relationship struct {
	FromID   string            // ID of the source memory
	ToID     string            // ID of the target memory
	Type     string            // Type of relationship
	Metadata map[string]string // Additional metadata about the relationship
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
		VectorDBDimensions:  3072, // For OpenAI embeddings
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

// injectMemories uses the Memory (if any) to fetch relevant context for the given message.
func (a *BaseAgent) injectMemories(ctx context.Context, message LLMMessage) LLMMessage {
	enhancedMessage := message
	if a.Memory != nil {
		if memoryEnhancer, ok := a.Memory.(MemoryEnhancer); ok {
			enhancedContext, err := memoryEnhancer.PrepareContext(ctx, a.Name, message.Content)
			if err == nil && enhancedContext != "" {
				output.Verbose(fmt.Sprintf("Enhanced input with memories (%d → %d chars)",
					len(message.Content), len(enhancedContext)))
				enhancedMessage.Content = enhancedContext
				errnie.Info(fmt.Sprintf("Enhanced input with %d characters of memories",
					len(enhancedContext)-len(message.Content)))
			} else if err != nil {
				output.Debug(fmt.Sprintf("Memory enhancement failed: %v", err))
			} else {
				output.Debug("No relevant memories found")
			}
		} else {
			output.Debug("Memory system does not support context enhancement")
		}
	} else {
		output.Debug("No memory system available")
	}
	return enhancedMessage
}

// extractMemories uses the Memory (if any) to extract new memories from the conversation text.
func (a *BaseAgent) extractMemories(ctx context.Context, contextWindow string) {
	if a.Memory != nil {
		if memoryExtractor, ok := a.Memory.(MemoryExtractor); ok {
			output.Verbose("Extracting memories from conversation")

			memories, err := memoryExtractor.ExtractMemories(ctx, a.Name, contextWindow, "conversation")
			if err != nil {
				output.Error("Memory extraction failed", err)
				errnie.Error(err)
			} else if memories != nil {
				output.Result(fmt.Sprintf("Extracted %d memories", len(memories)))
			}
		} else {
			output.Debug("Memory system does not support memory extraction")
		}
	}
}
