# 🤖 Caramba

[![Go CI/CD](https://github.com/theapemachine/caramba/actions/workflows/main.yml/badge.svg)](https://github.com/theapemachine/caramba/actions/workflows/main.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/theapemachine/caramba)](https://goreportcard.com/report/github.com/theapemachine/caramba)
[![GoDoc](https://godoc.org/github.com/theapemachine/caramba?status.svg)](https://godoc.org/github.com/theapemachine/caramba)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=TheApeMachine_caramba&metric=bugs)](https://sonarcloud.io/summary/new_code?id=TheApeMachine_caramba)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=TheApeMachine_caramba&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=TheApeMachine_caramba)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=TheApeMachine_caramba&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=TheApeMachine_caramba)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=TheApeMachine_caramba&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=TheApeMachine_caramba)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=TheApeMachine_caramba&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=TheApeMachine_caramba)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=TheApeMachine_caramba&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=TheApeMachine_caramba)
[![Technical Debt](https://sonarcloud.io/api/project_badges/measure?project=TheApeMachine_caramba&metric=sqale_index)](https://sonarcloud.io/summary/new_code?id=TheApeMachine_caramba)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=TheApeMachine_caramba&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=TheApeMachine_caramba)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=TheApeMachine_caramba&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=TheApeMachine_caramba)

A specialized agent framework in Go.

## ✨ Features

### 🧠 Multi-Provider Intelligence

To avoid rate-limits as much as possible, and circumvent provider-based bias in the responses,
you can use the `BalancedProvider` which wraps the underlying model providers, and spreads out
individual generations among them.

- OpenAI (GPT-4)
- Anthropic (Claude)
- Google (Gemini)
- Cohere (Command-R)
- Smart load balancing and failover
- Automatic cooldown and recovery
- Provider health monitoring
- Thread-safe operations

### 🔄 Pipeline Architecture

- Sequential and parallel execution modes
- Stage-based processing
- Event-driven communication
- Thread-safe concurrent execution
- Custom output aggregation
- Graph-based agent orchestration
- Node and edge-based workflow management

### 🛠 Tool System

- JSON schema-based tool definition
- Dynamic tool registration
- Streaming tool execution
- Generic parameter handling
- IO connectivity for streaming tools
- Runtime tool execution
- Schema-based tool registration

### 📝 Context Management

- Token-aware message history (128k context window)
- Intelligent message truncation
- System and user message preservation
- Optimized token counting using `tiktoken-go`
- Thread-safe buffer operations
- Message history management
- Context window optimization

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

### Infrastructure Setup

#### Docker Compose

The project includes a comprehensive Docker Compose setup that provides all necessary infrastructure:

```bash
# Start the infrastructure
docker compose up -d

# Available Services:
# - Ollama (GPU-enabled LLM server) - Port 11434
# - MinIO (Object Storage) - Ports 9000, 9001
# - Qdrant (Vector Database) - Port 6333
# - Redis (Cache) - Port 6379
# - Neo4j (Graph Database) - Ports 7474, 7687
```

#### Environment Setup

Create a `.env` file in your project root with the following variables:

```bash
# AI Provider Keys
OPENAI_API_KEY="your-key"
ANTHROPIC_API_KEY="your-key"
GEMINI_API_KEY="your-key"
COHERE_API_KEY="your-key"
HF_API_KEY="your-key"

# Integration Keys
GITHUB_PAT="your-github-pat"
AZDO_ORG_URL="your-azure-devops-url"
AZDO_PAT="your-azure-devops-pat"
TRENGO_API_KEY="your-trengo-key"
MARVIN_APP_KEY="your-marvin-app-key"
MARVIN_BOT_KEY="your-marvin-bot-key"
MARVIN_USER_KEY="your-marvin-user-key"
NVIDIA_API_KEY="your-nvidia-key"

# Infrastructure Credentials
MINIO_USER="your-minio-user"
MINIO_PASSWORD="your-minio-password"
QDRANT_API_KEY="your-qdrant-key"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your-neo4j-password"
```

## 🏗 Architecture

The system follows a sophisticated multi-layered architecture:

### Message Flow

1. Input → Agent
2. Agent → Provider
3. Provider → Buffer
4. Buffer → Tool (if applicable)
5. Tool → Agent
6. Agent → Output

#### Thread Safety

- Mutex-based provider synchronization
- Thread-safe buffer operations
- Channel-based communication
- Concurrent pipeline execution

### Error Handling

- Provider failure tracking
- Cooldown periods for failed providers
- Error event propagation
- Graceful pipeline termination

### Performance

- Token counting optimization
- Message history truncation
- Parallel execution capabilities
- Buffer size management

## 🔧 Configuration

Caramba uses Viper for configuration management, supporting:

- Environment-based provider configuration
- System prompts
- Role definitions
- JSON schema validation

## 🧪 Testing

```bash
go test -v ./...
```

The project includes comprehensive tests using GoConvey for better test organization and readability.

## 🛣 Roadmap

### Monitoring

- [ ] Metrics collection
- [ ] Logging strategy
- [ ] Performance monitoring
- [ ] Provider health tracking
- [ ] System telemetry

### Scalability

- [ ] Distributed execution
- [ ] Rate limiting
- [ ] Caching layer
- [ ] Load balancing improvements
- [ ] Horizontal scaling support

### Reliability

- [ ] Circuit breakers
- [ ] Retry strategies
- [ ] Error recovery
- [ ] Failover mechanisms
- [ ] System resilience

## 📚 Documentation

For detailed documentation on specific components:

- [Agent System](docs/agent.md)
- [Provider Management](docs/providers.md)
- [Pipeline System](docs/pipeline.md)
- [Tool System](docs/tools.md)
- [Context Management](docs/context.md)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to all the AI providers and open-source projects that make Caramba possible.
