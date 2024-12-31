# Context Management

The Context Management system in Caramba handles message history, token counting, and context window optimization for effective AI interactions.

## Architecture

### Context Structure

```go
type Context struct {
    Thread     *provider.Thread
    Scratchpad *provider.Thread
    System     *System
    indent     int
}
```

### Message Structure

```go
type Message struct {
    Role    string
    Content string
}
```

## Features

### Token Management

-   128k context window support
-   Intelligent message truncation
-   Token counting optimization
-   Window size monitoring

### Message History

-   System message preservation
-   User message tracking
-   Assistant response management
-   Tool call recording

### Thread Safety

-   Mutex-based synchronization
-   Atomic operations
-   Race condition prevention
-   Safe state updates

### Optimization

-   Smart message pruning
-   Context window maximization
-   Token usage efficiency
-   Memory optimization

## Usage

### Basic Context Creation

```go
context := ai.NewContext(system, params)
context.Compile()
```

### Message Management

```go
// Add a message to the thread
thread.AddMessage(provider.NewMessage(
    provider.RoleUser,
    "Message content",
))

// Compile context for generation
params := context.Compile()
```

### Scratchpad Usage

```go
// Get a fresh scratchpad
scratchpad := context.GetScratchpad()

// Append to current message
scratchpad.Append(event)

// Handle tool calls
scratchpad.ToolCall(event)
```

## Advanced Features

### Message Roles

-   System: Configuration and behavior definition
-   User: Input messages and queries
-   Assistant: AI-generated responses
-   Tool: Tool execution results

### Context Compilation

-   Message ordering
-   Role assignment
-   Token counting
-   Window optimization

### State Management

-   Thread state tracking
-   Scratchpad management
-   System state preservation
-   Context window monitoring

## Best Practices

1. **Message Management**

    - Monitor message length
    - Preserve critical messages
    - Implement proper truncation
    - Track token usage

2. **Context Window**

    - Monitor window size
    - Implement smart pruning
    - Preserve important context
    - Optimize token usage

3. **Thread Safety**

    - Use proper synchronization
    - Handle concurrent access
    - Prevent race conditions
    - Maintain state consistency

4. **Performance**
    - Optimize message storage
    - Implement efficient pruning
    - Monitor memory usage
    - Cache when appropriate

## Examples

### Basic Context Usage

```go
// Create a new context
context := ai.NewContext(system, params)

// Add messages
context.Thread.AddMessage(provider.NewMessage(
    provider.RoleUser,
    "What is the weather like?",
))

// Compile for generation
params := context.Compile()
```

### Advanced Context Management

```go
// Create a context with specific parameters
context := ai.NewContext(system, &provider.GenerationParams{
    MaxTokens: 4096,
    Temperature: 0.7,
})

// Add system message
context.Thread.AddMessage(provider.NewMessage(
    provider.RoleSystem,
    "You are a helpful assistant.",
))

// Add user message
context.Thread.AddMessage(provider.NewMessage(
    provider.RoleUser,
    "Help me with a task.",
))

// Compile and generate
params := context.Compile()
response := provider.Generate(ctx, params)
```

## Troubleshooting

Common issues and solutions:

1. **Context Overflow**

    - Monitor token count
    - Implement proper truncation
    - Preserve critical messages
    - Optimize message storage

2. **Memory Issues**

    - Monitor memory usage
    - Implement efficient pruning
    - Cache appropriately
    - Clean up unused context

3. **Performance Problems**
    - Optimize compilation
    - Implement efficient storage
    - Use appropriate data structures
    - Monitor resource usage

## Future Development

Planned improvements:

-   Enhanced token management
-   Advanced context optimization
-   Improved memory efficiency
-   Extended window support
-   Performance enhancements
-   Advanced pruning strategies
