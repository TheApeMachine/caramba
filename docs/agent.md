# Agent System

The Agent System is the core component of Caramba, responsible for coordinating AI interactions and executing complex tasks.

## Overview

Agents in Caramba are sophisticated entities that combine:

-   System prompts for behavior definition
-   Identity management for tracking and persistence
-   Context management for maintaining conversation state
-   Tool integration for extended capabilities
-   Provider interaction for AI model access

## Components

### Identity

```go
type Identity struct {
    Name string
    Role string
}
```

-   Provides trackable parameters for agent identification
-   Persists agent state across sessions
-   Enables role-based behavior modification

### System

-   Manages system prompts and instructions
-   Controls agent behavior and capabilities
-   Supports both structured and unstructured outputs

### Context

-   Maintains message history with 128k context window
-   Implements intelligent message truncation
-   Preserves critical system and user messages
-   Optimizes token usage with tiktoken-go

## Usage

### Basic Agent Creation

```go
agent := ai.NewAgent(ctx, "researcher", 3)
agent.Initialize()
```

### Adding Tools

```go
agent.AddTools(
    tools.NewBrowser(),
    tools.NewContainer(),
    // Add other tools as needed
)
```

### Message Generation

```go
message := provider.NewMessage(provider.RoleUser, "Query text...")
response := agent.Generate(context.Background(), message)

for event := range response {
    switch event.Type {
    case provider.EventChunk:
        // Handle text chunk
    case provider.EventToolCall:
        // Handle tool call
    case provider.EventError:
        // Handle error
    }
}
```

## Advanced Features

### Provider Management

-   Smart load balancing across multiple AI providers
-   Automatic failover and recovery
-   Health monitoring and cooldown periods
-   Thread-safe operations

### Tool Integration

-   Dynamic tool registration and discovery
-   JSON schema-based tool definition
-   Streaming tool execution support
-   Generic parameter handling

### Error Handling

-   Graceful error recovery
-   Comprehensive error reporting
-   Automatic retry mechanisms
-   Context preservation during failures

## Best Practices

1. **Initialization**

    - Always initialize agents before use
    - Configure appropriate tools based on agent role
    - Set reasonable iteration limits

2. **Context Management**

    - Monitor context window usage
    - Implement proper message truncation
    - Preserve critical system messages

3. **Tool Usage**

    - Register only necessary tools
    - Handle tool errors appropriately
    - Implement proper cleanup

4. **Provider Configuration**
    - Configure multiple providers when possible
    - Set appropriate timeouts
    - Monitor provider health

## Examples

### Research Agent

```go
agent := ai.NewAgent(ctx, "researcher", 5)
agent.AddTools(tools.NewBrowser(), tools.NewQdrantStore("research", 1536))
agent.Initialize()
```

### Development Agent

```go
agent := ai.NewAgent(ctx, "developer", 3)
agent.AddTools(tools.NewContainer(), tools.NewGithub())
agent.Initialize()
```

## Troubleshooting

Common issues and their solutions:

1. **Context Overflow**

    - Implement proper message truncation
    - Monitor token usage
    - Clear context when appropriate

2. **Tool Failures**

    - Check tool initialization
    - Verify required credentials
    - Monitor tool execution timeouts

3. **Provider Issues**
    - Verify API keys
    - Check provider status
    - Monitor rate limits

## Future Development

Planned improvements:

-   Enhanced context management
-   Advanced tool orchestration
-   Improved provider balancing
-   Extended error recovery mechanisms
