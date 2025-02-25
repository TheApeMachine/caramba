# Memory System

The memory system in Caramba provides agents with both short-term and long-term memory capabilities, enabling them to store information and retrieve it when needed.

## Overview

The memory system is designed with three key components:

1. **Base Memory**: Simple key-value storage for basic memory needs
2. **Vector Store**: Semantic search capabilities through vector embeddings
3. **Graph Store**: Relationship-based memory organization

Memory can be personal (belonging to a specific agent) or global (shared across agents).

## Features

- **Unified Memory Interface**: All memory types implement the core `Memory` interface
- **Vector-based Memory**: Semantic search using embeddings (via QDrant)
- **Graph-based Memory**: Relationship-based memory storage (via Neo4j)
- **Memory Extraction**: Automatically extract important information worth remembering
- **Context Preparation**: Enrich prompts with relevant memories before agent execution
- **Relationships**: Create connections between related memories in the graph

## Components

### UnifiedMemory

The main memory system that combines all memory types:

```go
memory := memory.NewUnifiedMemory(baseStore, embeddingProvider, options)
```

### EnhancedMemoryEntry

Enhanced memory entries with rich metadata:

```go
type EnhancedMemoryEntry struct {
    ID          string                 // Unique identifier for the memory
    AgentID     string                 // ID of the agent who owns the memory
    Content     string                 // The actual memory content
    Embedding   []float32              // Vector embedding of the content
    Type        MemoryType             // Type of memory (personal/global)
    Source      string                 // Source of the memory
    CreatedAt   time.Time              // When the memory was created
    AccessCount int                    // How many times this memory has been accessed
    LastAccess  time.Time              // When this memory was last accessed
    Metadata    map[string]interface{} // Additional metadata
}
```

### Memory Types

Memory can be categorized as:

- `MemoryTypePersonal`: Private to a specific agent
- `MemoryTypeGlobal`: Shared across all agents

### Memory Stores

- `InMemoryStore`: Simple in-memory key-value store
- `QDrantStore`: Vector-based store for semantic search
- `Neo4jStore`: Graph-based store for relationships

## Usage

### Basic Memory Operations

```go
// Store a memory
memory.Store(ctx, "key", "value")

// Retrieve a memory
value, err := memory.Retrieve(ctx, "key")

// Search for memories
results, err := memory.Search(ctx, "query", 5)

// Clear all memories
memory.Clear(ctx)
```

### Enhanced Memory Operations

```go
// Store a memory with metadata
memoryID, err := memory.StoreMemory(
    ctx,
    "agent-1",                  // Agent ID
    "Memory content",           // Content
    memory.MemoryTypePersonal,  // Type
    "conversation",             // Source
    map[string]interface{}{     // Metadata
        "importance": 0.8,
    },
)

// Retrieve memories by vector similarity
memories, err := memory.RetrieveMemoriesByVector(
    ctx,
    "search query",  // Query
    "agent-1",       // Agent ID (optional)
    5,               // Result limit
    0.7,             // Similarity threshold
)

// Create a relationship between memories
err := memory.CreateRelationship(ctx, memory.Relationship{
    FromID:   "memory-1",
    ToID:     "memory-2",
    Type:     "related_to",
    Metadata: nil,
})

// Get related memories
related, err := memory.GetRelatedMemories(
    ctx,
    "memory-1",      // Memory ID
    "related_to",    // Relationship type (optional)
    2,               // Max depth
)

// Extract memories from text
extracted, err := memory.ExtractMemories(
    ctx,
    "agent-1",       // Agent ID
    "Text content",  // Content to extract from
    "document",      // Source
)

// Prepare context with relevant memories
enrichedContext, err := memory.PrepareContext(
    ctx,
    "agent-1",       // Agent ID
    "original query" // Original query
)
```

## Integration with Agents

To use memory with an agent:

```go
// Create the memory system
baseStore := memory.NewInMemoryStore()
embeddingProvider := memory.NewOpenAIEmbeddingProvider(apiKey, "")
memoryOptions := memory.DefaultUnifiedMemoryOptions()
unifiedMemory, _ := memory.NewUnifiedMemory(baseStore, embeddingProvider, memoryOptions)

// Create an agent with memory
agent := core.NewBaseAgent("agent-with-memory")
agent.SetMemory(unifiedMemory)

// Execute with memory-enhanced context
response, err := agent.Execute(ctx, "What do you know about water?")
```

## Best Practices

1. **Identify Important Information**: Be selective about what to store in long-term memory
2. **Use Appropriate Memory Types**: Personal for agent-specific info, global for shared knowledge
3. **Create Relationships**: Connect related memories for better retrieval
4. **Consider Memory Decay**: Implement strategies to "forget" less important or outdated information
5. **Tune Similarity Thresholds**: Adjust thresholds based on your application needs
