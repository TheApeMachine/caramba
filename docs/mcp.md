# Model Context Protocol (MCP) in Caramba

The Model Context Protocol (MCP) is a standard defining how AI models (like Claude, or Caramba's internal LLM interactions) can discover and interact with external tools. Caramba leverages MCP in two main ways:

1. **Internal Tool Usage:** Caramba's agents use the MCP format to understand and request the execution of available tools (like `Browser`, `Memory`, `GitHub`, etc.) when interacting with LLM providers that support it.
2. **External MCP Server (Optional):** Caramba can optionally run as an MCP server itself, exposing its registered tools to external MCP-compatible clients (e.g., the Claude console, other AI frameworks).

This document focuses on the tools Caramba provides and how they are exposed via MCP when Caramba runs as an MCP server.

## Overview of Caramba as an MCP Server

When configured to run as an MCP server, Caramba:

- Exposes a specific API endpoint (defined in configuration) that adheres to the MCP specification.
- Allows MCP clients to request a list of available tools and their schemas (parameters, descriptions).
- Accepts requests from clients to execute specific tools with given parameters.
- Executes the tool logic internally (often using the `Generate` pattern with `datura.Artifact`).
- Returns the tool execution results (or errors) to the client in the standard MCP format.

## Available Tools via MCP

When Caramba runs as an MCP server, the following built-in tools (from `pkg/tools/`) can typically be made available:

1. **Memory Tool (`memory_...`)**

    - Operations: `memory_add`, `memory_query_vector`, `memory_query_graph`, etc.
    - Integrates with configured QDrant (vector) and/or Neo4j (graph) instances.
    - Handles persistent memory storage and retrieval.

2. **Editor Tool (`editor_...`)**

    - Operations: `editor_read_file`, `editor_write_file`, `editor_apply_diff`, etc.
    - Provides file system interaction capabilities.
    - Manages reading, writing, and modifying files.

3. **GitHub Tool (`github_...`)**

    - Operations: `github_create_pr`, `github_get_issue`, `github_list_files`, etc.
    - Handles integrations with GitHub repositories (requires PAT).
    - Manages repository operations like pull requests, issues, file access.

4. **Azure Tool (`azure_...`)**

    - Operations: `azure_create_work_item`, `azure_get_wiki_page`, etc.
    - Integrates with Azure DevOps (requires PAT).
    - Manages work items and wiki pages.

5. **Browser Tool (`browser_...`)**

    - Operations: `browser_get_content`, `browser_navigate`, `browser_run_javascript`, etc.
    - Provides web interaction capabilities via headless browsing.
    - Handles page navigation, content extraction, and script execution.

6. **Environment Tool (`environment_...`)**

    - Operations: `environment_run_command`.
    - Manages terminal command execution within a controlled environment.

7. **Slack Tool (`slack_...`)**

    - Operations: `slack_post_message`, `slack_search_messages`, `slack_add_reaction`, etc.
    - Integrates with Slack workspaces (requires Bot Token).

8. **Trengo Tool (`trengo_...`)**
    - Handles Trengo-specific operations (details depend on implementation).

*(Note: The exact list of available tools and their specific MCP operation names depend on which tools are registered when the MCP service is started. The `AI Agent` mentioned previously is not an MCP tool itself, but rather the entity that *uses* these tools internaly).*

## Usage

### Starting Caramba as an MCP Server

The exact method depends on the current command-line interface or configuration structure. It might involve:

- A specific command: e.g., `./caramba mcp-serve`
- A flag: e.g., `./caramba --mode mcp`
- A configuration setting in `config.yml`.

Consult `./caramba --help` or the main application setup (`cmd/caramba/main.go` or similar) for the precise way to start the MCP server.

**Conceptual Go Snippet (Illustrative):**

```go
// This is a conceptual example of how the MCP service might be started internally.
// You typically wouldn't write this code yourself, but run the compiled Caramba binary.

package main

import (
 "log"

 "github.com/theapemachine/caramba/pkg/service"
 "github.com/theapemachine/caramba/pkg/tools"
 // ... other necessary imports like config loaders ...
)

func main() {
 // 1. Load Configuration (which tools to enable, API keys, server settings)
 // ... config loading logic ...

 // 2. Create Tool Registry and Register Enabled Tools
 registry := tools.NewRegistry()

 // Example: Registering Browser and Memory tools if enabled in config
 if config.Tools.Browser.Enabled {
  registry.Register(tools.NewBrowser()) // Assuming NewBrowser returns the *MCP* tool wrapper
 }
 if config.Tools.Memory.Enabled {
        // MemoryTool might require db clients configured from env/config
        qdrantClient := memory.NewQdrant(/* config */)
        neo4jClient := memory.NewNeo4j(/* config */)
  registry.Register(tools.NewMemoryTool(qdrantClient, neo4jClient))
 }
 // ... register other configured tools (GitHub, Slack, etc.) ...

 // 3. Create and Start the MCP Service
 // The service needs the registry to know which tools to expose.
 mcpService := service.NewMCP(registry) // Pass registry to the service

 log.Println("Starting Caramba MCP Service...")
 err := mcpService.Start() // This blocks or runs in the background, listening for MCP requests
 if err != nil {
  log.Fatalf("Failed to start MCP service: %v", err)
 }

    // Keep the application running if Start() doesn't block
    // select{}
}
```

### MCP Server Configuration

Key aspects usually configured (in `config.yml` or similar):

- **Host/Port:** Where the MCP server listens (e.g., `localhost:8081`).
- **Enabled Tools:** Which of the available tools should be exposed.
- **Authentication:** How clients authenticate to the MCP server (if required).
- **Resource Limits:** Potential limits on tool execution.

### Tool Execution Flow (External Client Perspective)

1. **Client:** Connects to Caramba's MCP endpoint.
2. **Client:** (Optional) Requests the list of available tools (`mcp_GetTools`).
3. **Caramba (MCP Server):** Returns the list of registered tools and their schemas.
4. **Client:** Sends a request to execute a specific tool (`mcp_CallTool`) with parameters.
5. **Caramba (MCP Server):**
    - Receives the request.
    - Finds the corresponding tool in its registry.
    - Invokes the tool's internal logic (often the `Generate` method, passing parameters via a `datura.Artifact`).
    - Waits for the tool's result artifact.
    - Formats the result (or error) into an MCP `CallToolResult`.
6. **Caramba (MCP Server):** Sends the `CallToolResult` back to the client.

*(The internal use of `datura.Artifact` and `Generate` is abstracted away from the external MCP client, which only sees the standard MCP request/response format.)*

## Security

- **Authentication:** Secure your MCP endpoint appropriately if exposing it externally.
- **Tool Permissions:** Consider fine-grained control over which clients can use which tools (though standard MCP has limited support for this).
- **Resource Limits:** Protect the server from resource exhaustion due to intensive tool operations.

## Best Practices for Integration

- **Use Schemas:** Rely on the tool schemas provided by `mcp_GetTools` to correctly format parameters for `mcp_CallTool`.
- **Error Handling:** Check the `type` field in `CallToolResult` to distinguish between successful results (`text`, `json`) and errors (`error`).
- **Timeouts:** Implement appropriate timeouts on the client-side for tool calls.
