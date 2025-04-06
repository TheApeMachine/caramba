# Examples

## Basic AI Agent

Create a simple AI agent that can interact with various tools:

```go
package main

import (
    "context"
    "fmt"
    "io"
    "os"

    "github.com/theapemachine/caramba/pkg/ai"
    "github.com/theapemachine/caramba/pkg/core"
    "github.com/theapemachine/caramba/pkg/datura"
    "github.com/theapemachine/caramba/pkg/errnie"
    "github.com/theapemachine/caramba/pkg/provider"
    "github.com/theapemachine/caramba/pkg/tools"
)

func main() {
    // Create context for cancellation
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // Initialize agent
    agent := ai.NewAgentBuilder(
        ai.WithCancel(ctx),
        ai.WithIdentity("assistant", "helper"),
        ai.WithProvider(provider.ProviderTypeOpenAI),
        ai.WithParams(ai.NewParamsBuilder(
            ai.WithModel("gpt-4o"),
            ai.WithTemperature(0.7),
        )),
        ai.WithContext(ai.NewContextBuilder(
            ai.WithMessages(
                ai.NewMessageBuilder(
                    ai.WithRole("user"),
                    ai.WithContent("Hello, what can you help me with?"),
                ),
            ),
        )),
    )

    // Create streamer to handle communication
    streamer := core.NewStreamer(agent)

    // Create artifact with initial message
    artifact := datura.New(
        datura.WithEncryptedPayload([]byte("Hello, can you help me with a task?")),
    )

    // Process through streamer and display response
    if _, err := io.Copy(streamer, artifact); err != nil {
        errnie.Error(err)
        return
    }

    // Output response
    if _, err := io.Copy(os.Stdout, streamer); err != nil {
        errnie.Error(err)
    }
}
```

## Web Research Agent

Create an agent that can browse the web and process information:

```go
// Initialize agent with browser tool
agent := ai.NewAgentBuilder(
    ai.WithCancel(ctx),
    ai.WithIdentity("researcher", "web-explorer"),
    ai.WithProvider(provider.ProviderTypeOpenAI),
    ai.WithParams(ai.NewParamsBuilder(
        ai.WithModel("gpt-4o"),
        ai.WithTemperature(0.5),
    )),
    ai.WithTools(
        tools.NewToolBuilder(tools.WithMCP(tools.NewBrowser().Schema.ToMCP())),
    ),
    ai.WithContext(ai.NewContextBuilder(
        ai.WithMessages(
            ai.NewMessageBuilder(
                ai.WithRole("user"),
                ai.WithContent("Research the latest developments in AI and summarize them."),
            ),
        ),
    )),
)

// Create streamer
streamer := core.NewStreamer(agent)

// Process request and get response
io.Copy(streamer, datura.New())
io.Copy(os.Stdout, streamer)
```

## Memory-Enhanced Agent

Create an agent with long-term memory capabilities:

```go
// Initialize agent with memory tools
agent := ai.NewAgentBuilder(
    ai.WithCancel(ctx),
    ai.WithIdentity("memory-agent", "assistant"),
    ai.WithProvider(provider.ProviderTypeOpenAI),
    ai.WithParams(ai.NewParamsBuilder(
        ai.WithModel("gpt-4o"),
        ai.WithTemperature(0.7),
    )),
    ai.WithTools(
        tools.NewToolBuilder(tools.WithMCP(tools.NewMemoryTool().Schema.ToMCP())),
    ),
    ai.WithContext(ai.NewContextBuilder(
        ai.WithMessages(
            ai.NewMessageBuilder(
                ai.WithRole("user"),
                ai.WithContent("What do you remember about our previous conversations?"),
            ),
        ),
    )),
)

// Create streamer
streamer := core.NewStreamer(agent)

// Process request and get response
io.Copy(streamer, datura.New())
io.Copy(os.Stdout, streamer)
```

## System Interaction Agent

Create an agent that can interact with the system environment:

```go
// Initialize agent with environment tools
agent := ai.NewAgentBuilder(
    ai.WithCancel(ctx),
    ai.WithIdentity("system-agent", "admin"),
    ai.WithProvider(provider.ProviderTypeOpenAI),
    ai.WithParams(ai.NewParamsBuilder(
        ai.WithModel("gpt-4o"),
        ai.WithTemperature(0.3),
    )),
    ai.WithTools(
        tools.NewToolBuilder(tools.WithMCP(tools.NewEnvironment().Schema.ToMCP())),
        tools.NewToolBuilder(tools.WithMCP(tools.NewSystemInspectTool().Schema.ToMCP())),
        tools.NewToolBuilder(tools.WithMCP(tools.NewSystemOptimizeTool().Schema.ToMCP())),
    ),
    ai.WithContext(ai.NewContextBuilder(
        ai.WithMessages(
            ai.NewMessageBuilder(
                ai.WithRole("user"),
                ai.WithContent("List all running Docker containers and their status."),
            ),
        ),
    )),
)

// Create streamer
streamer := core.NewStreamer(agent)

// Process request and get response
io.Copy(streamer, datura.New())
io.Copy(os.Stdout, streamer)
```

## Multi-Provider Example

Create an example that can switch between AI providers:

```go
// Function to create an agent with specified provider
func createAgent(providerType provider.ProviderType, model string) *ai.AgentBuilder {
    return ai.NewAgentBuilder(
        ai.WithCancel(ctx),
        ai.WithIdentity("agent", "assistant"),
        ai.WithProvider(providerType),
        ai.WithParams(ai.NewParamsBuilder(
            ai.WithModel(model),
            ai.WithTemperature(0.7),
        )),
        ai.WithContext(ai.NewContextBuilder(
            ai.WithMessages(
                ai.NewMessageBuilder(
                    ai.WithRole("user"),
                    ai.WithContent("Compare how different models handle this prompt."),
                ),
            ),
        )),
    )
}

// Create agents with different providers
openaiAgent := createAgent(provider.ProviderTypeOpenAI, "gpt-4o")
anthropicAgent := createAgent(provider.ProviderTypeAnthropic, "claude-3-opus-20240229")
googleAgent := createAgent(provider.ProviderTypeGoogle, "gemini-pro")

// Create streamers
openaiStreamer := core.NewStreamer(openaiAgent)
anthropicStreamer := core.NewStreamer(anthropicAgent)
googleStreamer := core.NewStreamer(googleAgent)

// Process with each provider
io.Copy(openaiStreamer, datura.New())
io.Copy(anthropicStreamer, datura.New())
io.Copy(googleStreamer, datura.New())

// Output results
fmt.Println("OpenAI Response:")
io.Copy(os.Stdout, openaiStreamer)
fmt.Println("\nAnthropic Response:")
io.Copy(os.Stdout, anthropicStreamer)
fmt.Println("\nGoogle Response:")
io.Copy(os.Stdout, googleStreamer)
```
