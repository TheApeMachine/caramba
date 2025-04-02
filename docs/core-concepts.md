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
agent := ai.NewAgentBuilder(
    ai.WithIdentity("assistant", "helper"),
    ai.WithProvider(provider.ProviderTypeOpenAI),
    ai.WithParams(ai.NewParamsBuilder(
        ai.WithModel("gpt-4o"),
        ai.WithTemperature(0.7),
    )),
    ai.WithTools(
        tools.NewToolBuilder(tools.WithMCP(tools.NewBrowser().Schema.ToMCP())),
        tools.NewToolBuilder(tools.WithMCP(tools.NewMemoryTool().Schema.ToMCP())),
    ),
)
```

### Providers

Providers are the AI service integrations that:

- Handle API communication
- Manage streaming responses
- Process tool calls
- Format structured outputs

The system supports multiple providers:

```go
// Available provider types
provider.ProviderTypeMock
provider.ProviderTypeOpenAI
provider.ProviderTypeAnthropic
provider.ProviderTypeGoogle
provider.ProviderTypeCohere
provider.ProviderTypeDeepSeek
provider.ProviderTypeOllama
```

### Tools

Tools extend agent capabilities through a unified interface:

- Browser for web automation
- Memory for data persistence
- Environment for system interaction
- Editor for file manipulation
- Azure DevOps integration
- GitHub integration
- Slack integration
- Custom tools through MCP

```go
// Tool creation
tools.NewToolBuilder(tools.WithMCP(tools.NewBrowser().Schema.ToMCP()))
tools.NewToolBuilder(tools.WithMCP(tools.NewEditor().Schema.ToMCP()))
tools.NewToolBuilder(tools.WithMCP(tools.NewMemoryTool().Schema.ToMCP()))
```

## Stream Processing

Caramba implements a stream-based architecture:

- IO-based data flow
- Generator pattern for data processing
- Artifact-based communication
- Buffer management

```go
// Creating and using streamers
streamer := core.NewStreamer(agent)
io.Copy(streamer, artifact)
```

### Artifacts

Artifacts are secure data containers:

- Structured message format
- Metadata management
- Serialization support
- Cross-component communication

```go
artifact := datura.New(
    datura.WithPayload(data),
    datura.WithMetadata(meta),
)
```

## Memory Integration

Long-term storage through:

- QDrant for vector storage
- Neo4j for graph relationships
- Unified memory interface
- Context persistence

```go
// Memory components
memory.NewQdrant()
memory.NewNeo4j()
```

## MCP Integration

Model Context Protocol support:

- Tool integration with AI models
- Standardized interfaces
- Client-server architecture
- Bidirectional communication

```go
// Starting MCP server
service := service.NewMCP()
service.Start()
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

### Secure Artifacts

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

### Persistent Storage

Long-term storage through:

- QDrant for vector storage
- Neo4j for graph relationships
- Unified memory interface
- Context persistence
