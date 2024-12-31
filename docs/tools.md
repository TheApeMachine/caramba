# Tool System

The Tool System in Caramba provides a flexible and extensible framework for integrating various capabilities into agents through a unified interface.

## Architecture

### Tool Interface

```go
type Tool interface {
    GenerateSchema() interface{}
    Use(context.Context, map[string]any) string
    Connect(context.Context, io.ReadWriteCloser) error
}
```

## Available Tools

### Browser Tool

```go
type Browser struct {
    Operation  string // navigate, script
    URL        string
    Javascript string
}
```

Features:

-   Headless browser automation
-   Stealth mode support
-   Proxy configuration
-   JavaScript execution
-   Screenshot capability

### Container Tool

```go
type Container struct {
    Reasoning string
    Command   string
}
```

Features:

-   Isolated Debian environment
-   Command execution
-   Interactive shell
-   File system access
-   Resource management

### Database Tools

#### Neo4j Tool

```go
type Neo4jQuery struct {
    Cypher    string
    Reasoning string
}

type Neo4jStore struct {
    Cypher    string
    Reasoning string
}
```

Features:

-   Graph database operations
-   Cypher query execution
-   Relationship management
-   Data persistence

#### Qdrant Tool

```go
type QdrantQuery struct {
    Query     string
    Reasoning string
}

type QdrantStore struct {
    Documents []string
    Metadata  map[string]any
    Reasoning string
}
```

Features:

-   Vector similarity search
-   Document storage
-   Metadata management
-   Semantic querying

### Integration Tools

#### Azure Tool

```go
type Azure struct {
    Operation string // search, create_ticket, get_tickets, update_ticket
}
```

Features:

-   Cloud service operations
-   Ticket management
-   Resource search
-   Status updates

#### GitHub Tool

```go
type Github struct {
    Repo      string
    Operation string // clone, pull, push
}
```

Features:

-   Repository operations
-   Version control
-   Code management
-   Collaboration support

#### Slack Tool

```go
type Slack struct {
    Operation string // send_message, search
    Channel   string
    Message   string
}
```

Features:

-   Message sending
-   Channel management
-   Search functionality
-   Event handling

#### Trengo Tool

```go
type Trengo struct {
    Operation string // search, add_labels, get_labels
}
```

Features:

-   Customer communication
-   Label management
-   Search functionality
-   Integration support

## Usage

### Tool Registration

```go
agent.AddTools(
    tools.NewBrowser(),
    tools.NewContainer(),
    tools.NewNeo4j(),
    tools.NewQdrant("collection", 1536),
)
```

### Tool Execution

```go
// Browser example
browserTool := tools.NewBrowser()
result := browserTool.Use(ctx, map[string]any{
    "operation": "navigate",
    "url": "https://example.com",
    "javascript": "() => document.title",
})

// Container example
containerTool := tools.NewContainer()
result = containerTool.Use(ctx, map[string]any{
    "command": "ls -la",
})
```

## Best Practices

1. **Tool Selection**

    - Choose appropriate tools for tasks
    - Consider resource implications
    - Handle tool dependencies
    - Manage tool lifecycle

2. **Error Handling**

    - Implement proper error checking
    - Handle tool failures gracefully
    - Provide meaningful error messages
    - Implement recovery mechanisms

3. **Resource Management**

    - Clean up tool resources
    - Monitor resource usage
    - Implement timeouts
    - Handle concurrent access

4. **Security**
    - Validate tool inputs
    - Manage credentials securely
    - Implement access controls
    - Monitor tool usage

## Advanced Features

### Custom Tool Creation

```go
type CustomTool struct {
    // Tool-specific fields
}

func (ct *CustomTool) GenerateSchema() interface{} {
    return utils.GenerateSchema[*CustomTool]()
}

func (ct *CustomTool) Use(ctx context.Context, params map[string]any) string {
    // Implementation
}

func (ct *CustomTool) Connect(ctx context.Context, conn io.ReadWriteCloser) error {
    // Implementation
}
```

### Tool Chaining

```go
// Example of chaining browser and database tools
browserResult := browserTool.Use(ctx, browserParams)
qdrantTool.Use(ctx, map[string]any{
    "documents": []string{browserResult},
    "metadata": map[string]any{"source": "web"},
})
```

## Troubleshooting

Common issues and solutions:

1. **Connection Issues**

    - Check network connectivity
    - Verify credentials
    - Monitor timeouts
    - Check resource availability

2. **Resource Constraints**

    - Monitor memory usage
    - Check disk space
    - Handle concurrent access
    - Implement resource limits

3. **Tool Failures**
    - Check error messages
    - Verify input parameters
    - Monitor tool state
    - Implement retries

## Future Development

Planned improvements:

-   Enhanced tool discovery
-   Advanced tool orchestration
-   Improved error handling
-   Extended tool capabilities
-   Performance optimizations
-   Additional tool integrations
