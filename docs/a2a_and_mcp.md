# A2A Agents as MCP Resources

This document describes how Agent-to-Agent (A2A) protocol agents are implemented as Model Context Protocol (MCP) resources in the Caramba framework.

## Overview

The A2A specification states that A2A agents can be implemented as MCP resources, represented by their agent cards. This allows MCP clients to discover and interact with A2A agents through the standard MCP resource interface.

In our implementation:

1. A2A agents are registered with a central `Catalog`
2. An `AgentResourceManager` exposes these agents as MCP resources using the `agent://` URI scheme
3. The MCP server registers these resources, making them available to MCP clients

## Implementation Details

### Agent Cards

Agent cards contain metadata about an agent, including:

- Name
- Description
- URL (endpoint where the agent can be reached)
- Provider information
- Capabilities
- Skills

These cards are defined in the `pkg/agent/card.go` file and represent an A2A agent's identity and capabilities.

### Resource URIs

Agent resources are exposed using the `agent://` URI scheme, with the agent name as the path:

```
agent://{agent_name}
```

For example, an agent named "chat_agent" would have the URI `agent://chat_agent`.

### Resource Manager

The `AgentResourceManager` in `pkg/resources/agent_manager.go` implements the `ResourceManager` interface and is responsible for:

1. Listing available agents as resources
2. Reading agent card content as JSON
3. Supporting subscription to agent updates (when an agent's capabilities change)

### Registration with MCP

When an agent is registered with the `MCPServer` via the `RegisterAgent` method, it:

1. Adds the agent to the catalog
2. Creates an MCP resource for the agent
3. Adds a handler that returns the agent card as a JSON resource
4. Makes the agent discoverable via the standard `resources/list` endpoint

## Usage Example

The `examples/chat.go` file demonstrates how to register an A2A agent with the MCP server:

```go
// Create a new MCP server with A2A support
srv := service.NewMCPServer()

// Register an A2A agent
chatAgent := &catalog.Agent{
    Name:        "chat_agent",
    Description: "A terminal-based chat agent that supports A2A protocol",
    URL:         "http://localhost:8080",
}

// Register the agent with the MCP server to expose it as a resource
srv.RegisterAgent(chatAgent)

// Start the MCP server
go func() {
    if err := srv.Start(); err != nil {
        errnie.Fatal(err)
    }
}()
```

MCP clients can then discover and access the agent via the resource URI `agent://chat_agent`.

## Benefits

This integration of A2A and MCP offers several advantages:

1. **Discoverability**: MCP clients can automatically discover available A2A agents
2. **Standardized Access**: Agents are exposed through a well-defined, consistent interface
3. **Metadata Access**: Clients can retrieve agent metadata (capabilities, skills) without direct agent interaction
4. **Real-time Updates**: Using the subscription mechanism, clients can be notified when agents change

## Future Enhancements

Future improvements to this implementation could include:

1. Adding agent skill templates to enable more dynamic discovery
2. Supporting agent task subscription through MCP resource subscription
3. Enhanced agent authentication and authorization mechanisms
4. Multi-agent coordination through MCP resource relationships
