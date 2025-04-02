# 🤖 Caramba

A powerful, Go-based agent framework that follows the "everything is `io`" philosophy. Connect anything to anything with ease through a unified interface where all objects implement at least `io.ReadWriteCloser`.

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

## MCP

Caramba can be used as an MCP (Model Context Protocol) server, providing built-in tools to AI clients like Claude. This feature allows external AI models to access Caramba's tools and capabilities.

First, build the binary:

```bash
go build
```

Start the MCP server:

```bash
caramba mcp
```

To integrate with Claude or other MCP clients, modify the config:

```json
"mcpServers": {
    "caramba": {
      "command": "/path/to/caramba mcp",
      "args": [],
      "env": {
        "PATH": "/your/path/environment",
        "AZURE_DEVOPS_ORG": "<your azure devops org>",
        "AZURE_DEVOPS_ORG_URL": "<your azure devops org url>",
        "AZDO_PAT": "<your azure devops personal access token>",
        "AZURE_DEVOPS_PROJECT": "<your azure devops project>",
        "SLACK_DEFAULT_CHANNEL": "<your slack channel id>",
        "SLACK_BOT_TOKEN": "<your slack bot token>",
        "GITHUB_PAT": "<your github personal access token>",
        "OPENAI_API_KEY": "<your OpenAI key>"
      }
    }
}
```

## ✨ Features

Caramba provides a unified I/O interface with various components that work together seamlessly.

### Tool System

- [x] Model Context Protocol (MCP) integration
      [Read More](docs/mcp.md)
- [x] Agent with dynamic tool loading
- [x] Browser automation for web interaction
- [x] Memory integration
  - [x] QDrant vector store
  - [x] Neo4j graph database
- [x] Docker integration
- [x] Kubernetes integration
- [x] GitHub integration
- [x] Azure DevOps integration
- [x] File Editor with code manipulation
- [x] Environment tool for terminal interaction
- [x] Slack integration

### Stream Processing

- [x] IO-based architecture
- [x] Bidirectional data flow
- [x] Error handling and propagation
- [x] Buffer management
      [Read More](pkg/stream/README.md)

### AI Provider Support

- [x] OpenAI
  - [x] Streaming responses
  - [x] Tool calling
  - [x] Function calling
  - [x] Structured outputs
  - [x] Embeddings
- [x] Anthropic (Claude)
  - [x] Streaming responses
  - [x] Tool calling
- [x] Google (Gemini)
- [x] Cohere
- [x] DeepSeek
- [x] Ollama (for local models)

### System Components

- [x] Error handling (errnie)
- [x] File system abstraction
- [x] Service management
- [x] Configuration management
- [x] Docker/Kubernetes orchestration

## 🚀 Quick Start

### Prerequisites

- Go 1.21 or higher
- Docker (optional, for running QDrant, Neo4j, etc.)
- API keys for desired AI providers

### Installation

```bash
go get github.com/theapemachine/caramba
```

### Basic Usage

Build the binary:

```bash
go build
```

Run the MCP server:

```bash
./caramba mcp
```

Run an example:

```bash
./caramba example
```

## 📚 Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Core Concepts](docs/core-concepts.md)
- [MCP Integration](docs/mcp.md)
- [Examples](docs/examples.md)

## 🛠️ Example: Creating an Agent

```go
// Initialize an AI agent with tools
agent := ai.NewAgent(
    ai.WithProvider("openai"),
    ai.WithTools(
        tools.NewBrowser(),
        tools.NewEditor(),
        tools.NewMemory(),
    ),
)

// Use the agent to interact with AI
response, err := agent.Ask("Look up the current weather and save it to weather.txt")
if err != nil {
    log.Fatal(err)
}

fmt.Println(response)
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](docs/contributing.md) for details on how to submit pull requests and report issues.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
