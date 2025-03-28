# Model Context Protocol (MCP)

The Model Context Protocol (MCP) is a powerful feature in Caramba that provides a standardized interface for AI model interactions and tool management. It allows seamless integration between different components while maintaining a unified I/O approach.

## Overview

MCP in Caramba runs as a service that supports both standard I/O and Server-Sent Events (SSE) communication methods. It provides:

- Tool management and execution
- Resource capabilities
- Prompt handling
- Bidirectional communication

## Available Tools

The MCP service comes with several built-in tools:

1. **Memory Tool**

   - Integrates with QDrant vector store
   - Connects to Neo4j graph database
   - Handles persistent memory operations

2. **AI Agent**

   - Manages AI model interactions
   - Handles tool orchestration

3. **Editor Tool**

   - Provides file editing capabilities
   - Manages code modifications

4. **Github Tool**

   - Handles GitHub integrations
   - Manages repository operations

5. **Azure Tool**

   - Integrates with Azure services
   - Manages Azure DevOps operations

6. **Trengo Tool**

   - Handles Trengo-specific operations

7. **Browser Tool**

   - Provides web interaction capabilities
   - Handles browser automation

8. **Environment Tool**
   - Manages terminal interactions
   - Handles system operations

## Usage

### Starting the MCP Server

```go
service := service.NewMCP()
err := service.Start()
```

### Server Configuration

The MCP server is initialized with:

- Server name: "caramba-server"
- Version: "1.0.0"
- Resource capabilities enabled
- Prompt capabilities enabled
- Tool capabilities enabled

### SSE Configuration

For SSE connections:

- Base URL: [http://localhost:8080](http://localhost:8080)
- Authentication: Handled through request headers

## Tool Execution

Tools in MCP follow a unified execution pattern:

1. Tools implement the `io.ReadWriteCloser` interface
2. Data is passed through artifacts with specific roles
3. Results are returned in a standardized format

Example tool execution flow:

```go
artifact := datura.New(
    datura.WithRole(role),
    datura.WithMeta(key, value)
)

// Copy data to tool
io.Copy(tool, artifact)

// Get result from tool
io.Copy(buffer, tool)
```

## Security

- Authentication is handled through request headers
- Context-based security model
- Secure tool execution environment

## Error Handling

The MCP service includes comprehensive error handling:

- Tool execution errors are captured and returned
- System-level errors are logged and managed
- Debug-level logging available for troubleshooting

## Best Practices

1. Always check tool execution results
2. Handle authentication properly
3. Use appropriate artifact roles for different operations
4. Implement proper error handling
5. Monitor tool performance and resource usage

## Integration

To integrate with the MCP service:

1. Connect to the appropriate endpoint (stdio or SSE)
2. Authenticate if required
3. Use the available tools through the standardized interface
4. Handle responses appropriately

## Future Enhancements

The MCP system is designed to be extensible. New tools can be added by:

1. Implementing the `io.ReadWriteCloser` interface
2. Creating appropriate schemas
3. Registering with the MCP service
