# Getting Started with Caramba

This guide will help you get up and running with Caramba, the Go-based agent framework.

## Installation

Install Go (1.21 or higher)
Clone the repository:

```bash
git clone https://github.com/theapemachine/caramba.git
cd caramba
```

Build the binary:

```bash
go build
```

## Environment Setup

Create a `.env` file in your project root with your API keys:

```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
AZURE_DEVOPS_PAT=your_token_here
GITHUB_PAT=your_token_here
SLACK_BOT_TOKEN=your_token_here
```

## Using Caramba as an MCP Server

Start the MCP server:

```bash
./caramba mcp
```

This launches Caramba as a Model Context Protocol server, allowing AI systems like Claude to access its tools.

## Basic Usage Example

### Creating a Simple Agent

```go
package main

import (
    "context"
    "io"
    "os"

    "github.com/theapemachine/caramba/pkg/ai"
    "github.com/theapemachine/caramba/pkg/core"
    "github.com/theapemachine/caramba/pkg/datura"
    "github.com/theapemachine/caramba/pkg/provider"
    "github.com/theapemachine/caramba/pkg/tools"
)

func main() {
    // Create context for cancellation
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // Create an agent
    agent := ai.NewAgentBuilder(
        ai.WithCancel(ctx),
        ai.WithIdentity("assistant", "helper"),
        ai.WithProvider(provider.ProviderTypeOpenAI),
        ai.WithParams(ai.NewParamsBuilder(
            ai.WithModel("gpt-4o"),
            ai.WithTemperature(0.7),
        )),
        ai.WithTools(
            tools.NewToolBuilder(tools.WithMCP(tools.NewBrowser().Schema.ToMCP())),
        ),
    )

    // Create a message
    message := ai.NewMessageBuilder(
        ai.WithRole("user"),
        ai.WithContent("What is the current weather in New York?"),
    )

    // Add message to context
    context := ai.NewContextBuilder(
        ai.WithMessages(message),
    )

    // Update agent with context
    ai.WithContext(context)(agent)

    // Create streamer
    streamer := core.NewStreamer(agent)

    // Process request and output response
    io.Copy(streamer, datura.New())
    io.Copy(os.Stdout, streamer)
}
```

### Running Built-in Examples

Caramba comes with built-in examples:

```bash
# Run the code generation example
./caramba example
```

## Next Steps

- Read about [Core Concepts](core-concepts.md)
- Learn about MCP integration in [MCP Documentation](mcp.md)
- Check out more [Examples](examples.md)
- Contribute to the project (see [GitHub repository](https://github.com/theapemachine/caramba))
