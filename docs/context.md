# Context Management

The Context Management system in Caramba handles message history, token counting, and context window optimization for effective AI interactions.

## Architecture

### Context Structure

```go
type Context struct {
    Identity  *Identity
    Toolcalls []*provider.Event
    indent    int
}
```

The Context structure now focuses on identity-based management and tool call tracking, with the following components:

- `Identity`: Manages agent identity and system prompts
- `Toolcalls`: Tracks tool interactions during conversation
- `indent`: Handles structural formatting

### Identity Management

```go
type Identity struct {
    System string
    Params *provider.LLMGenerationParams
}
```

The Identity component manages:

- System prompts and instructions
- Generation parameters
- Thread management
- Agent behavior definition

### Message Structure

```go
type Message struct {
    Role    string
    Content string
}
```

## Features

### Token Management

- 128k context window support
- Intelligent message truncation
- Token counting optimization
- Window size monitoring

### Message History

- System message preservation
- User message tracking
- Assistant response management
- Tool call recording

### Thread Safety

- Mutex-based synchronization
- Atomic operations
- Race condition prevention
- Safe state updates

### Optimization

- Smart message pruning
- Context window maximization
- Token usage efficiency
- Memory optimization

### Quick Context Creation

```go
// Create a context with specific steering instructions
ctx := QuickContext(
    systemPrompt,
    "codeswitch",  // Optional steering additions
    "noexplain",
    "silentfail",
)
```

### Iteration Management

```go
// Compile context for a specific iteration
params := ctx.Compile(currentCycle, maxIterations)

// Reset context state
ctx.Reset()

// Add new message
ctx.AddMessage(provider.NewMessage(
    provider.RoleUser,
    "Message content",
))
```

## Usage

### Basic Context Creation

```go
// Create a new context with identity
identity := NewIdentity(ctx, "reasoner", systemPrompt)
context := NewContext(identity)

// Add messages and compile
context.AddMessage(provider.NewMessage(
    provider.RoleUser,
    "What is the weather like?",
))
params := context.Compile(0, 3) // First iteration of max 3
```

### Advanced Context Management

```go
// Create a context with steering instructions
context := QuickContext(
    systemPrompt,
    "codeswitch",
    "noexplain",
)

// Add messages
context.AddMessage(provider.NewMessage(
    provider.RoleUser,
    "Help me with a task.",
))

// Get string representation
contextStr := context.String(false) // Exclude system messages

// Reset context
context.Reset()
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

- System: Configuration and behavior definition
- User: Input messages and queries
- Assistant: AI-generated responses and iteration markers
- Tool: Tool execution results and tracking

### Context Compilation

- Message ordering and formatting
- Iteration cycle management
- System prompt inclusion
- Tool call tracking

### State Management

- Identity-based configuration
- Tool call history
- System prompt preservation
- Iteration state tracking

### Steering Instructions

Available steering options:

- `codeswitch`: Modify response style
- `noexplain`: Reduce explanatory content
- `silentfail`: Handle failures quietly
- `scratchpad`: Enable working memory

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

- Enhanced token management
- Advanced context optimization
- Improved memory efficiency
- Extended window support
- Performance enhancements
- Advanced pruning strategies
