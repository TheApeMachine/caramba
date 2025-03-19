# Feedback

The `Feedback` type is a crucial component in Caramba's workflow system that enables bidirectional data flow between components in a pipeline. At its core, it utilizes `io.TeeReader` to create a "tee" connection that can split data streams into multiple directions.

## Overview

The primary purpose of the `Feedback` type is to enable components to communicate in both directions within a pipeline. This is particularly useful when you need to:

- Feed responses from an AI provider back to an agent
- Create loops in your data processing workflow
- Monitor or log data flowing through a pipeline
- Implement reactive patterns where components need to adapt based on downstream results

## Implementation

The `Feedback` type consists of three main components:

- `forward`: An `io.ReadWriter` that receives and processes data moving forward in the pipeline
- `backward`: An `io.Writer` that receives a copy of the data for reverse flow
- `tee`: An `io.Reader` created using `io.TeeReader` that manages the splitting of data

## Usage Example

Here's a typical usage pattern in an AI workflow:

```go
agent := ai.NewAgent(...)
provider := provider.NewOpenAIProvider(...)

pipeline := workflow.NewPipeline(
    agent,
    workflow.NewFeedback(provider, agent), // Connect provider's output back to agent
    workflow.NewConverter(),
)
```

In this example:

1. The agent sends a request to the provider
2. The provider processes the request and generates a response
3. The Feedback component ensures the response flows both:
   - Forward through the pipeline to the converter (which in the original example code streams the response to the terminal, providing real-time generation results)
   - Backward to the agent to maintain context

## Best Practices

- Use Feedback when components need to maintain state or context based on downstream results
- Consider the memory implications when teeing large data streams
- Ensure both forward and backward components properly implement the required interfaces
- Close the Feedback component when done to properly cleanup resources

The Feedback type is a cornerstone of Caramba's "everything is `io`" philosophy, making it easy to create sophisticated workflows while maintaining a clean and consistent interface.
