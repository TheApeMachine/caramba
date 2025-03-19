# Core Concepts

## Everything is I/O

The fundamental principle of Caramba is that everything implements `io.ReadWriteCloser`. This design choice enables:

- Seamless component connection
- Unified data flow
- Composable workflows
- Bidirectional communication

## Components

### Agents

Agents are the core processing units that:

- Handle messages and tool calls
- Maintain conversation context
- Execute actions through tools
- Process responses

```go
agent := ai.NewAgent(
    ai.WithModel("gpt-4"),
    ai.WithTools(tools),
)
```

### Providers

Providers are the AI service integrations that:

- Handle API communication
- Manage streaming responses
- Process tool calls
- Format structured outputs

```go
provider := provider.NewOpenAIProvider(
    os.Getenv("OPENAI_API_KEY"),
    endpoint,
)
```

### Tools

Tools extend agent capabilities:

- Browser for web interaction
- Memory for data persistence
- Environment for system interaction
- Custom tools through MCP

```go
tools := []*provider.Tool{
    tools.NewBrowser().Schema,
    tools.NewMemoryTool(stores).Schema,
}
```

## Workflow System

### Pipelines

Pipelines connect components:

- Sequential data flow
- Component chaining
- Error handling
- Resource management

```go
pipeline := workflow.NewPipeline(
    agent,
    workflow.NewFeedback(provider, agent),
    workflow.NewConverter(),
)
```

### Feedback Loops

Feedback enables bidirectional flow:

- Response processing
- Context updates
- State management
- Error recovery

```go
feedback := workflow.NewFeedback(
    provider,
    agent,
)
```

## Data & Security

### Artifacts

Artifacts are secure data containers:

- Encrypted payloads
- Metadata management
- Cryptographic signatures
- Versioning support

```go
artifact := datura.New(
    datura.WithPayload(data),
    datura.WithMetadata(meta),
)
```

### Memory Integration

Long-term storage through:

- QDrant for vector storage
- Neo4j for graph relationships
- Unified memory interface
- Context persistence

## Stream Processing

Built-in streaming support:

- Chunked data handling
- Real-time processing
- Resource efficiency
- Backpressure handling
