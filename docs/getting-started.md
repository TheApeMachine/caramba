# Getting Started with Caramba

This guide will help you get up and running with Caramba, the Go-based agent framework.

## Installation

1. Install Go (1.21 or higher)
2. Install Caramba:

```bash
go get github.com/theapemachine/caramba
```

## Environment Setup

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
# Add other provider keys as needed
```

Start required services:

```bash
docker compose up
```

## Basic Usage

### 1. Create a Simple Chat Agent

```go
package main

import (
    "github.com/theapemachine/caramba/pkg/ai"
    "github.com/theapemachine/caramba/pkg/provider"
    "github.com/theapemachine/caramba/pkg/workflow"
)

func main() {
    // Initialize agent
    agent := ai.NewAgent(
        ai.WithModel("gpt-4"),
    )

    // Setup provider
    provider := provider.NewOpenAIProvider(
        os.Getenv("OPENAI_API_KEY"),
        "https://api.openai.com/v1",
    )

    // Create workflow
    pipeline := workflow.NewPipeline(
        agent,
        workflow.NewFeedback(provider, agent),
        workflow.NewConverter(),
    )

    // Use the pipeline...
}
```

### 2. Run Examples

Caramba comes with built-in examples:

```bash
# Run the pipeline example
go run main.go examples pipeline

# Run the chat example
go run main.go examples chat
```

## Next Steps

- Read about [Core Concepts](core-concepts.md)
- Explore the [API Reference](api-reference.md)
- Check out more [Examples](examples.md)
- Learn how to [Contribute](contributing.md)
