# Memory Package

The memory package provides the implementation of memory capabilities for the Caramba agent framework. It allows agents to store, retrieve, and semantically search information from past interactions.

## Features

- **Basic Memory Storage**: Simple key-value storage for agent memories
- **Semantic Search**: Find relevant memories based on semantic similarity
- **LLM-based Memory Extraction**: Use LLMs to extract important information from conversations
- **Vector Database Integration**: Store and retrieve memories using vector embeddings
- **Memory Enhancement**: Enhance agent inputs with relevant memories from past interactions

## Components

### Memory Interfaces

The core interfaces are defined in the agent core package:

- `Memory`: Base interface for storing and retrieving memories
- `MemoryEnhancer`: Interface for enhancing agent inputs with relevant memories
- `MemoryExtractor`: Interface for extracting important information from text

### Implementations

1. **BaseMemory**: Simple in-memory implementation of the Memory interface
2. **LLMMemoryExtractor**: Uses an LLM to extract important memories from text
3. **QDrantMemory**: Vector-based memory implementation using QDrant (simulated)

### Embedding Providers

- **OpenAIEmbeddings**: Generate vector embeddings using OpenAI's embedding models

## Usage

### Basic Memory

```go
// Create a simple in-memory storage
memory := memory.NewBaseMemory()

// Store a memory
err := memory.Store(ctx, "user_preference", "User prefers dark mode")

// Retrieve a memory
value, err := memory.Retrieve(ctx, "user_preference")
```

### LLM Memory Extraction

```go
// Create an LLM-based memory extractor
llm := openai.NewProvider(apiKey, "gpt-4")
memoryExtractor := memory.NewLLMMemoryExtractor(llm)

// Extract memories from a conversation
conversation := "User: Can you help me with setting up my Go project?\nAgent: Of course, I'd be happy to help with your Go project setup."
memories, err := memoryExtractor.ExtractMemories(ctx, "assistant", conversation, "conversation")
```

### Semantic Search with QDrant

```go
// Set up OpenAI embeddings provider
embeddings := memory.NewOpenAIEmbeddings(apiKey, "")

// Create QDrant memory
qdrantMemory := memory.NewQDrantMemory(embeddings, "agent_memories")

// Store memories with embeddings
qdrantMemory.Store(ctx, "fact_1", "The capital of France is Paris")

// Search for relevant memories
results, err := qdrantMemory.Search(ctx, "Tell me about France", 5)
```

## Integration with Agents

The memory system is designed to be used with the Caramba agent framework. Once attached to an agent, it will:

1. Store user inputs and agent responses automatically
2. Extract important information from conversations
3. Enhance future inputs with relevant memories

```go
agent := core.NewBaseAgent()
agent.SetLLM(llm)
agent.SetMemory(qdrantMemory)

// The agent will now use memory during execution
response, err := agent.Execute(ctx, "Hello, can you help me with my project?")
```
