# 🤖 Caramba

A sophisticated multi-agent AI orchestration system built in Go, designed to coordinate multiple AI providers and facilitate complex reasoning tasks through a pipeline-based architecture.

## ✨ Features

### 🧠 Multi-Provider Intelligence

-   OpenAI (GPT-4)
-   Anthropic (Claude)
-   Google (Gemini)
-   Cohere (Command)
-   Smart load balancing and failover
-   Automatic cooldown and recovery
-   Provider health monitoring
-   Thread-safe operations

### 🔄 Pipeline Architecture

-   Sequential and parallel execution modes
-   Stage-based processing
-   Event-driven communication
-   Thread-safe concurrent execution
-   Custom output aggregation
-   Graph-based agent orchestration
-   Node and edge-based workflow management

### 🛠 Tool System

-   JSON schema-based tool definition
-   Dynamic tool registration
-   Streaming tool execution
-   Generic parameter handling
-   IO connectivity for streaming tools
-   Runtime tool execution
-   Schema-based tool registration

#### Available Tools

-   **Browser Tool**: Headless browser automation with stealth mode, proxy support, and JavaScript execution
-   **Container Tool**: Isolated Debian environment for command execution
-   **Database Tools**:
    -   Neo4j: Graph database querying and storage
    -   Qdrant: Vector database for similarity search and document storage
-   **Integration Tools**:
    -   Azure: Cloud service operations and ticket management
    -   GitHub: Repository operations (clone, pull, push)
    -   Slack: Message sending and channel management
    -   Trengo: Customer communication platform integration
-   **Tool Features**:
    -   Automatic schema generation
    -   Context-aware execution
    -   Error handling and recovery
    -   Streaming response support
    -   Concurrent operation capability

### 📝 Context Management

-   Token-aware message history (128k context window)
-   Intelligent message truncation
-   System and user message preservation
-   Optimized token counting using tiktoken-go
-   Thread-safe buffer operations
-   Message history management
-   Context window optimization

## 🚀 Quick Start

```bash
# Install Caramba
go get github.com/theapemachine/caramba

# Set up your environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export COHERE_API_KEY="your-key"
```

### Basic Usage

```go
package main

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
)

func main() {
    // Create a new agent
    agent := ai.NewAgent()

    // Send a message
    message := provider.NewMessage(provider.RoleUser, "Analyze this text...")
    response := agent.Generate(context.Background(), message)

    // Process the response
    for event := range response {
        fmt.Print(event.Text)
    }
}
```

## 🏗 Architecture

The system follows a sophisticated multi-layered architecture:

#### Message Flow

1. Input → Agent
2. Agent → Provider
3. Provider → Buffer
4. Buffer → Tool (if applicable)
5. Tool → Agent
6. Agent → Output

#### Thread Safety

-   Mutex-based provider synchronization
-   Thread-safe buffer operations
-   Channel-based communication
-   Concurrent pipeline execution

#### Error Handling

-   Provider failure tracking
-   Cooldown periods for failed providers
-   Error event propagation
-   Graceful pipeline termination

#### Performance

-   Token counting optimization
-   Message history truncation
-   Parallel execution capabilities
-   Buffer size management

## 🔧 Configuration

Caramba uses Viper for configuration management, supporting:

-   Environment-based provider configuration
-   System prompts
-   Role definitions
-   JSON schema validation

## 🧪 Testing

```bash
go test -v ./...
```

The project includes comprehensive tests using GoConvey for better test organization and readability.

## 🛣 Roadmap

### Monitoring

-   [ ] Metrics collection
-   [ ] Logging strategy
-   [ ] Performance monitoring
-   [ ] Provider health tracking
-   [ ] System telemetry

### Scalability

-   [ ] Distributed execution
-   [ ] Rate limiting
-   [ ] Caching layer
-   [ ] Load balancing improvements
-   [ ] Horizontal scaling support

### Reliability

-   [ ] Circuit breakers
-   [ ] Retry strategies
-   [ ] Error recovery
-   [ ] Failover mechanisms
-   [ ] System resilience

## 📚 Documentation

For detailed documentation on specific components:

-   [Agent System](docs/agent.md)
-   [Provider Management](docs/providers.md)
-   [Pipeline System](docs/pipeline.md)
-   [Tool System](docs/tools.md)
-   [Context Management](docs/context.md)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to all the AI providers and open-source projects that make Caramba possible.
