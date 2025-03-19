# Examples

## Basic Chat Agent

Create a simple chat agent that can interact with users:

```go
package main

import (
    "github.com/theapemachine/caramba/pkg/ai"
    "github.com/theapemachine/caramba/pkg/provider"
    "github.com/theapemachine/caramba/pkg/workflow"
)

func main() {
    // Create agent with OpenAI
    agent := ai.NewAgent(
        ai.WithModel("gpt-4"),
    )

    // Setup provider
    provider := provider.NewOpenAIProvider(
        os.Getenv("OPENAI_API_KEY"),
        tweaker.GetEndpoint("openai"),
    )

    // Create pipeline
    pipeline := workflow.NewPipeline(
        agent,
        workflow.NewFeedback(provider, agent),
        workflow.NewConverter(),
    )

    // Handle chat
    for {
        // Read user input
        fmt.Print("> ")
        input, _ := reader.ReadString('\n')

        // Process through pipeline
        msg := datura.New(
            datura.WithPayload(provider.NewParams(
                provider.WithMessages(
                    provider.NewMessage(
                        provider.WithUserRole("User", input),
                    ),
                ),
            ).Marshal()),
        )

        io.Copy(pipeline, msg)
        io.Copy(os.Stdout, pipeline)
    }
}
```

## Web Research Agent

Create an agent that can browse the web and process information:

```go
agent := ai.NewAgent(
    ai.WithModel("gpt-4"),
    ai.WithTools([]*provider.Tool{
        tools.NewBrowser().Schema,
    }),
)

provider := provider.NewOpenAIProvider(
    os.Getenv("OPENAI_API_KEY"),
    tweaker.GetEndpoint("openai"),
)

pipeline := workflow.NewPipeline(
    agent,
    workflow.NewFeedback(
        provider,
        agent,
    ),
    workflow.NewConverter(),
)

// Research task
msg := datura.New(
    datura.WithPayload(provider.NewParams(
        provider.WithModel("gpt-4"),
        provider.WithTools(tools.NewBrowser().Schema),
        provider.WithMessages(
            provider.NewMessage(
                provider.WithUserRole(
                    "User",
                    "Research the latest developments in AI and summarize them.",
                ),
            ),
        ),
    ).Marshal()),
)

io.Copy(pipeline, msg)
io.Copy(os.Stdout, pipeline)
```

## Memory-Enhanced Agent

Create an agent with long-term memory capabilities:

```go
// Setup memory stores
stores := map[string]io.ReadWriteCloser{
    "qdrant": memory.NewQdrant(),
    "neo4j":  memory.NewNeo4j(),
}

agent := ai.NewAgent(
    ai.WithModel("gpt-4"),
    ai.WithTools([]*provider.Tool{
        tools.NewMemoryTool(stores).Schema,
    }),
)

provider := provider.NewOpenAIProvider(
    os.Getenv("OPENAI_API_KEY"),
    tweaker.GetEndpoint("openai"),
)

pipeline := workflow.NewPipeline(
    agent,
    workflow.NewFeedback(provider, agent),
    workflow.NewConverter(),
)

// Query with memory context
msg := datura.New(
    datura.WithPayload(provider.NewParams(
        provider.WithMessages(
            provider.NewMessage(
                provider.WithUserRole(
                    "User",
                    "What do you remember about our previous conversations?",
                ),
            ),
        ),
    ).Marshal()),
)

io.Copy(pipeline, msg)
io.Copy(os.Stdout, pipeline)
```

## Environment Interaction

Create an agent that can interact with the system:

```go
agent := ai.NewAgent(
    ai.WithModel("gpt-4"),
    ai.WithTools([]*provider.Tool{
        tools.NewEnvironment().Schema,
    }),
)

provider := provider.NewOpenAIProvider(
    os.Getenv("OPENAI_API_KEY"),
    tweaker.GetEndpoint("openai"),
)

pipeline := workflow.NewPipeline(
    agent,
    workflow.NewFeedback(provider, agent),
    workflow.NewConverter(),
)

// System task
msg := datura.New(
    datura.WithPayload(provider.NewParams(
        provider.WithMessages(
            provider.NewMessage(
                provider.WithUserRole(
                    "User",
                    "List all running Docker containers and their status.",
                ),
            ),
        ),
    ).Marshal()),
)

io.Copy(pipeline, msg)
io.Copy(os.Stdout, pipeline)
```

## Multi-Provider Pipeline

Create a pipeline that uses multiple AI providers:

```go
openai := provider.NewOpenAIProvider(
    os.Getenv("OPENAI_API_KEY"),
    tweaker.GetEndpoint("openai"),
)

anthropic := provider.NewAnthropicProvider(
    os.Getenv("ANTHROPIC_API_KEY"),
    tweaker.GetEndpoint("anthropic"),
)

agent := ai.NewAgent(
    ai.WithModel("gpt-4"),
)

pipeline := workflow.NewPipeline(
    agent,
    workflow.NewFeedback(openai, agent),
    workflow.NewFeedback(anthropic, agent),
    workflow.NewConverter(),
)

// Process with both providers
msg := datura.New(
    datura.WithPayload(provider.NewParams(
        provider.WithMessages(
            provider.NewMessage(
                provider.WithUserRole(
                    "User",
                    "Compare how different models handle this prompt.",
                ),
            ),
        ),
    ).Marshal()),
)

io.Copy(pipeline, msg)
io.Copy(os.Stdout, pipeline)
```
