# Core Concepts

This document outlines the fundamental building blocks and design principles of the Caramba framework.

## Everything is I/O (Conceptually)

While not every component strictly implements `io.ReadWriteCloser` directly anymore, the _spirit_ of unified data flow remains. Caramba leverages interfaces and patterns (like the `Generate` pattern using channels and artifacts) to enable:

- Standardized component interaction
- Composable agent workflows
- Clear data pathways, often using `datura.Artifact`
- Bidirectional communication where needed (e.g., streaming)

## Components

### Task Handlers (e.g., `pkg/agent/handlers`)

These are the central orchestrators for incoming A2A requests. They:

- Receive tasks (`TaskCreate`, `TaskSend`, etc.) via the API layer.
- Manage task state using the `TaskStore` (`pkg/task`).
- Prepare parameters and context for the LLM `Provider`.
- Invoke the appropriate `Provider` to interact with the LLM.
- Handle streaming responses and events from the `Provider`.
- Coordinate tool execution requests originating from the LLM.
- Send status updates via SSE and/or push notifications.

### Providers (`pkg/provider`)

Providers are the bridge between Caramba and specific Large Language Models (LLMs) or AI services. They:

- Abstract the specifics of different LLM APIs (OpenAI, Anthropic, Google, Cohere, Ollama, etc.).
- Handle request formatting, including messages, system prompts, and tool definitions (MCP format).
- Manage API calls, including handling streaming responses.
- Parse LLM responses, identifying text generation, tool call requests, and errors.
- Emit standardized `ProviderEvent` types for task handlers to process.

```go
// Example: Getting a provider instance (Actual usage involves configuration)
import "github.com/theapemachine/caramba/pkg/provider"
import "github.com/theapemachine/caramba/pkg/tweaker" // For config access

// Configuration determines the actual provider
providerName := tweaker.GetProvider() // Reads from config (e.g., "openai")
llmProvider, err := provider.New(providerName)
if err != nil {
    // Handle error
}

// Parameters for the provider call
params := provider.ProviderParams{
    Model:      tweaker.GetModel(providerName), // e.g., "gpt-4o"
    Messages:   []provider.Message{ /* ... conversation history ... */ },
    Tools:      []mcp.Tool{ /* ... available tools ... */ },
    Stream:     true,
    // ... other LLM parameters (temp, top_p, etc.)
}

// Initiate generation (returns an event channel)
providerEventChan, err := llmProvider.Generate(params)
// ... process events from providerEventChan ...
```

### Tools (`pkg/tools`)

Tools extend the capabilities of the LLM agent, allowing it to interact with the outside world. They:

- Adhere to the **Model Context Protocol (MCP)** for standardized definition and invocation.
- Encapsulate specific functionalities: web browsing (`Browser`), file system operations (`Editor`, `fs`), memory access (`Memory`), environment interaction (`Environment`), external service APIs (GitHub, Slack, Azure DevOps, Trengo).
- Are registered in the `ToolRegistry` (`pkg/tools/registry.go`) for discovery.
- Often implement a `Generate(chan *datura.Artifact) chan *datura.Artifact` method to handle data flow using Caramba's artifact system.

```go
// Example: Interacting with a tool via the registry (Conceptual)
import (
    "context"
    "github.com/theapemachine/caramba/pkg/tools"
    "github.com/theapemachine/caramba/pkg/datura"
    "github.com/mark3labs/mcp-go/mcp"
)

// Assume toolRegistry is populated
var toolRegistry *tools.Registry

func callBrowserTool(url string) (string, error) {
    toolName := "browser_get_content" // Specific MCP name for the browser operation
    registeredTool, exists := toolRegistry.Get(toolName)
    if !exists {
        return "", fmt.Errorf("tool %s not found", toolName)
    }

    // Prepare MCP request (simplified example)
    mcpReq := mcp.CallToolRequest{
        ToolName: toolName,
        Parameters: map[string]any{
            "url": url,
            // Other browser params like "javascript", etc.
        },
    }

    // --- Option 1: Direct Use (Less common in Caramba's internal flow) ---
    /*
    mcpResult, err := registeredTool.Use(context.Background(), mcpReq)
    if err != nil {
        return "", err
    }
    if mcpResult.Type == mcp.ToolResultTypeError {
        return "", fmt.Errorf("tool error: %s", mcpResult.Content)
    }
    return mcpResult.Content, nil
    */

    // --- Option 2: Using the Generate pattern (More typical internal flow) ---
    inputArtifact := datura.New(
        // Set metadata needed by the tool's Generate method
        datura.WithMeta("operation", "get_content"),
        datura.WithMeta("url", url),
    )

    inputChan := make(chan *datura.Artifact, 1)
    inputChan <- inputArtifact
    close(inputChan)

    // The registeredTool.Generator is the actual tool implementation
    outputChan := registeredTool.Generator.Generate(inputChan)

    resultArtifact := <-outputChan // Wait for the result
    if resultArtifact.Error() != nil {
        return "", resultArtifact.Error()
    }

    payload, err := resultArtifact.Payload() // Use DecryptPayload if encrypted
    if err != nil {
        return "", err
    }
    return string(payload), nil
}

```

### Artifacts (`pkg/datura`)

Artifacts are the primary data carriers within Caramba. They provide a standardized way to pass information between components (handlers, tools, providers).

- **Secure:** Support encrypted payloads (`datura.WithEncryptedPayload`).
- **Rich Metadata:** Carry contextual information (`datura.WithMeta`, `datura.GetMetaValue`).
- **Role-Based:** Can indicate purpose (e.g., prompt, response, error) using `datura.WithRole`.
- **Traceable:** Can link artifacts via parent IDs (`datura.WithParent`).
- **Serializable:** Can be passed across processes or stored.

```go
import "github.com/theapemachine/caramba/pkg/datura"

// Create an artifact carrying a user prompt
promptText := "Summarize this document for me."
artifact := datura.New(
    datura.WithPayload([]byte(promptText)), // Use WithEncryptedPayload for sensitive data
    datura.WithRole(datura.ArtifactRolePrompt),
    datura.WithMeta("user_id", "user-123"),
    datura.WithMeta("document_url", "http://example.com/doc.txt"),
)

// Reading metadata
docURL := datura.GetMetaValue[string](artifact, "document_url")

// Reading payload
payloadBytes, err := artifact.Payload() // or DecryptPayload()
```

## Processing Flow (Simplified A2A Task)

1. **Client -> API:** Client sends `TaskCreate` request to `/rpc`.
2. **API -> Handler:** Request routed to `TaskCreate` handler in `pkg/agent/handlers`.
3. **Handler:** Creates task entry in `TaskStore` (`pkg/task`). Responds to client.
4. **Client -> API:** Client sends `TaskSend` request with the actual prompt.
5. **API -> Handler:** Request routed to `TaskSend` handler.
6. **Handler:** Retrieves task, prepares history, identifies available tools from `ToolRegistry`.
7. **Handler -> Provider:** Calls `provider.Generate` with messages, tools, and params.
8. **Provider -> LLM:** Provider formats request and calls the LLM API.
9. **LLM -> Provider:** LLM streams back response chunks (text, tool calls).
10. **Provider -> Handler:** Provider parses chunks and sends `ProviderEvent`s (e.g., `EventTypeChunk`, `EventTypeToolCall`) on the event channel.
11. **Handler:** Processes events:
    - Accumulates text chunks.
    - If tool call requested: prepares `datura.Artifact`, sends it to the appropriate tool's `Generate` method via the `ToolRegistry`.
    - Receives result artifact from the tool.
    - Formats tool result message.
    - Sends tool result back to `Provider` (often requires another `provider.Generate` call).
    - Sends SSE updates (`/task/:id/stream`) to the client.
12. **Handler:** Once LLM finishes (or error occurs), updates final task state in `TaskStore` and sends final SSE/notification.

## Memory Integration (`pkg/memory`)

Caramba agents can persist information using the `Memory` tool, which integrates with backend stores:

- **Vector Storage:** Typically uses QDrant (`memory.NewQdrant`) for semantic search over unstructured text.
- **Graph Storage:** Typically uses Neo4j (`memory.NewNeo4j`) for storing and querying structured relationships.
- **Unified Interface:** The `MemoryTool` (`pkg/tools/memory.go`) provides MCP methods (`memory_add`, `memory_query_vector`, `memory_query_graph`, etc.) that route requests to the appropriate backend client.

```go
// Example: Adding document to memory via MemoryTool (Conceptual MCP call)
// MCP Request Parameters:
// { "operation": "add", "document": "User prefers concise summaries." }

// Example: Querying vector memory
// MCP Request Parameters:
// { "operation": "query_vector", "question": "What are the user's preferences?" }

// Example: Querying graph memory
// MCP Request Parameters:
// { "operation": "query_graph", "cypher": "MATCH (u:User {id: 'user-123'})-[:PREFERS]->(p) RETURN p.name" }

// The MemoryTool implementation handles connecting to Qdrant/Neo4j based on the operation.
```

## MCP Integration (`pkg/service`, `pkg/tools/mcp.go`)

Beyond _using_ MCP to define tools for internal LLM interaction, Caramba can _act_ as an MCP server:

- **MCP Service:** (`pkg/service/mcp.go`) Can be started to expose Caramba's registered tools over an MCP-compliant API.
- **Use Case:** Allows external MCP clients (like AI models running elsewhere, e.g., Claude console) to discover and use Caramba's tools (Browser, GitHub, Memory, etc.).

```go
// Conceptual: Starting Caramba in MCP server mode
// (Actual mechanism might involve command-line flags or specific config)

// import "github.com/theapemachine/caramba/pkg/service"
// import "github.com/theapemachine/caramba/pkg/tools"

// func runMCP() {
//    registry := tools.NewRegistry()
//    registry.Register(tools.NewBrowser()) // Register desired tools
//    registry.Register(tools.NewMemoryTool())
//    // ... register other tools

//    mcpService := service.NewMCP(registry) // Pass registry to service
//    err := mcpService.Start() // Starts listening for MCP requests
//    if err != nil {
//        // handle error
//    }
// }
```

## Concurrency (`pkg/twoface`)

Caramba uses a worker pool system (`pkg/twoface`) to manage goroutines efficiently, preventing overload when handling many concurrent tasks or tool executions.

- **Pool:** Manages a dynamic set of worker goroutines.
- **Worker:** Executes individual jobs.
- **Job:** An interface for tasks that can be run by a worker.
- **Scaler:** Dynamically adjusts the number of workers based on load.

This is mostly an internal mechanism but ensures Caramba remains responsive under load.
