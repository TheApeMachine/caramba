# Getting Started with Caramba

Welcome! This guide provides a quick path to setting up and running your first Caramba agent. For more detailed explanations, check the main [README.md](../README.md).

## Prerequisites

- **Go:** Version 1.21 or higher. ([Installation Guide](https://go.dev/doc/install))
- **API Keys:** For LLM providers (OpenAI, Anthropic, etc.) and any tools requiring authentication (GitHub, Slack, Azure).
- **(Optional) Docker & Docker Compose:** If you plan to use tools requiring external services like vector databases (Qdrant) or graph databases (Neo4j) via the Memory tool.

## Installation & Setup

1. **Clone:**

    ```bash
    git clone https://github.com/theapemachine/caramba.git
    cd caramba
    ```

2. **Configure Environment:**

    - Create a `.env` file in the project root to store your API keys and other secrets. Caramba's main configuration (`cmd/cfg/config.yml`) often references these variables.

      ```env
      # .env (Example - Add keys/secrets you need)
      OPENAI_API_KEY="sk-..."
      ANTHROPIC_API_KEY="sk-..."
      GITHUB_PAT="ghp_..."
      # ... other keys/URIs as needed for providers and tools
      ```

    - Review and potentially adjust `cmd/cfg/config.yml`. This file controls agent identity, default LLM parameters, tool configurations, server settings, etc. Ensure it references your environment variables correctly (e.g., `openai_api_key: ${OPENAI_API_KEY}`).

3. **Build:**

    ```bash
    # Assuming your main package is cmd/caramba
    go build -o caramba ./cmd/caramba
    ```

    _(Adjust the path if your main package is located elsewhere)_

## Running Caramba

Execute the binary you just built:

```bash
./caramba
```

This starts the Caramba service, which includes:

- The A2A API endpoint (usually `http://localhost:8080/rpc` by default, check config/logs).
- The A2A Agent Card (`/.well-known/ai-agent.json`).
- The A2A streaming endpoint (`/task/:id/stream`).
- Potentially an MCP server endpoint if configured.

## Interacting with Your Agent (A2A Protocol)

You interact with the running Caramba agent by sending JSON-RPC requests according to the A2A protocol. Here's a basic example using `curl`:

```bash
# 1. Choose a Unique Task ID
TASK_ID="my-simple-task-$(date +%s)"
AGENT_URL="http://localhost:8080" # Adjust if needed

# 2. Create the Task
echo "Creating task $TASK_ID..."
curl -s -X POST "$AGENT_URL/rpc" \\
     -H "Content-Type: application/json" \\
     -d '{
       "jsonrpc": "2.0",
       "method": "TaskCreate",
       "params": {
         "task_id": "'$TASK_ID'",
         "input": {
           "request_id": "req-create-'$TASK_ID'",
           "messages": [
             {
               "role": "user",
               "parts": [ { "type": "text", "text": "Tell me a short story about a brave Go gopher." } ]
             }
           ]
         }
       },
       "id": "rpc-create-'$TASK_ID'"
     }' | cat # Use 'cat' or a JSON tool like 'jq' for readability

# 3. (Optional) Start listening for streaming updates in another terminal
# echo "Listening for updates on $AGENT_URL/task/$TASK_ID/stream"
# curl --no-buffer "$AGENT_URL/task/$TASK_ID/stream"

# 4. Send the Initial Message to Start Processing
echo "\\nSending message to task $TASK_ID..."
curl -s -X POST "$AGENT_URL/rpc" \\
     -H "Content-Type: application/json" \\
     -d '{
       "jsonrpc": "2.0",
       "method": "TaskSend",
       "params": {
         "task_id": "'$TASK_ID'",
         "message": {
           "request_id": "req-send-'$TASK_ID'",
           "role": "user",
           "parts": [ { "type": "text", "text": "Tell me a short story about a brave Go gopher." } ]
         },
         "subscribe": true
       },
       "id": "rpc-send-'$TASK_ID'"
     }' | cat # Use 'cat' or 'jq'

echo "\\nCheck the stream (if started) or wait for final status."
```

This script first creates a task placeholder using `TaskCreate`, then sends the actual prompt using `TaskSend`. If you uncomment the streaming `curl` command and run it, you'll see real-time updates.

## Using Caramba as an MCP Server (Optional)

If you need other systems (like Claude via its console) to use Caramba's tools, you might configure and run Caramba specifically as an MCP server. This often involves a different command or configuration flag (details depend on the current implementation - check `cmd/` or `./caramba --help` if available).

```bash
# Example (hypothetical command - verify!)
# ./caramba mcp-serve --config=mcp-config.yml
```

When run this way, Caramba exposes its registered tools (like Browser, Memory, GitHub) via the Model Context Protocol, allowing compatible AI models to invoke them.

## Running Built-in Examples (If Available)

Some projects include specific example commands. Check if Caramba has commands like:

```bash
# Example (hypothetical command - verify!)
# ./caramba run-example --name=code-generation
```

_(Refer to `./caramba --help` or the project structure for actual example commands)_

## Next Steps

- Dive deeper into [Core Concepts](core-concepts.md).
- Explore specific [Tool Capabilities](mcp.md) (Note: `mcp.md` might focus on the protocol, tool details are in `pkg/tools/`).
- See more complex scenarios in [Examples](examples.md) (Requires updating to new interaction patterns).
- Consult the main [README.md](../README.md) for architecture and detailed configuration.
- Contribute! See [Contributing Guidelines](contributing.md).
