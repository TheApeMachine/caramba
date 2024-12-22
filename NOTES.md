# Technical Implementation Notes

## Project Overview

Caramba is a multi-agent AI system implemented in Go, designed to coordinate multiple AI providers and facilitate complex reasoning tasks 
through a pipeline-based architecture.

## Core Components

### 1. Agent System (`ai/agent.go`)

- Implementation of individual AI agents

- Key features:
  - Supports multiple AI providers
  - Tool registration and execution
  - Implements `io.ReadWriteCloser` interface
  - Message buffering and context management
  - JSON-based tool execution

- Notable interfaces:

```go
type Agent struct {
    provider provider.Provider
    tools    map[string]Tool
    process  Process
    prompt   *Prompt
    buf      strings.Builder
    buffer   *Buffer
}
```

### 2. Provider Management (`provider/balanced.go`)

- Implements a balanced provider system for multiple AI services

- Supported providers:
  - OpenAI (GPT-4)
  - Anthropic (Claude)
  - Google (Gemini)
  - Cohere (Command)

- Features:
  - Provider health monitoring
  - Failure tracking and recovery
  - Load balancing
  - Cooldown periods
  - Thread-safe operations

### 3. Context Management (`ai/buffer.go`)

- Implements message history management

- Key features:
  - Token counting using tiktoken-go
  - Context window management (128k tokens)
  - Message truncation strategy
  - Preserves system and user messages

- Token estimation:
  - Uses OpenAI's token estimation guidelines
  - Reserves 500 tokens for responses

### 4. Pipeline System (`ai/pipeline.go`)

- Orchestrates multi-agent workflows

- Features:
  - Sequential and parallel execution modes
  - Stage-based processing
  - Custom output aggregation
  - Event-based communication
  - Thread-safe concurrent execution

### 5. Tool System (`ai/tool.go`)

- Interface for extensible tool implementation

```go
type Tool interface {
    GenerateSchema() string
    Use(map[string]any) string
    Connect(io.ReadWriteCloser)
}
```

- Features:
  - JSON schema-based tool definition
  - Generic parameter handling
  - IO connectivity for streaming tools

## Configuration System

- Uses Viper for configuration management
- Embedded default configuration
- Support for:
  - Environment-based provider configuration
  - System prompts
  - Role definitions
  - JSON schema validation

## Integration Points

1. Provider Integration:
   - API key configuration via environment variables
   - Model selection per provider
   - Error handling and retry logic

2. Tool Integration:
   - Schema-based tool registration
   - Runtime tool execution
   - IO streaming support

3. Process Integration:
   - Custom process registration
   - Schema generation
   - Role-based prompt construction

## Current Implementation State

### Completed Features

- Basic agent infrastructure
- Provider load balancing
- Context management
- Pipeline execution
- Tool interface definition
- Configuration system

### In Progress/TODO

- Documentation needs completion
- Error handling could be more robust
- Testing coverage needs expansion
- Some template text remains in root command

## Future Considerations

1. Monitoring:
   - Add metrics collection
   - Implement logging strategy
   - Add performance monitoring

2. Scalability:
   - Consider distributed execution
   - Implement rate limiting
   - Add caching layer

3. Reliability:
   - Implement circuit breakers
   - Add retry strategies
   - Improve error recovery

## Technical Decisions

### Message Flow

1. Input → Agent
2. Agent → Provider
3. Provider → Buffer
4. Buffer → Tool (if applicable)
5. Tool → Agent
6. Agent → Output

### Thread Safety

- Mutex-based synchronization for provider access
- Thread-safe buffer operations
- Concurrent pipeline execution
- Channel-based communication

### Error Handling

- Provider failure tracking
- Cooldown periods for failed providers
- Error event propagation
- Graceful pipeline termination

### Performance Considerations

- Token counting optimization
- Message history truncation
- Parallel execution when possible
- Buffer size management
