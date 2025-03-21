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

Besides an agent framework, Caramba can also be used as an MCP (Model Context Protocol) server, which will make all the built-in tools available to the client.

You will have to build a binary first.

```bash
go build
```

To run it as an `mcp` server, use the appropriate `cli` command.

```bash
caramba mcp
```

To integrate it with Claude, you will have to modify the config.

```json
"mcpServers": {
    ... (other MCP server configs),
    "caramba": {
      "command": "/path/to/caramba mcp",
      "args": [],
      "env": {
        "PATH": "It seems Claude does not have access to the ENV, so you can set your PATH this way",
        "AZURE_DEVOPS_ORG": "<your azure devops org>",
        "AZURE_DEVOPS_ORG_URL": "<your azure devops org url>",
        "AZDO_PAT": "<your azure devops personal access token>",
        "AZURE_DEVOPS_PROJECT": "<your azure devops project>",
        "SLACK_DEFAULT_CHANNEL": "<your slack channel id>",
        "SLACK_BOT_TOKEN": "<your slack bot token>",
        "GITHUB_PAT": "<your github personal access token>",
        "OPENAI_API_KEY": "<your OpenAI key>",
      }
    }
}
```

## ✨ Features

Caramba comes with a wide array of features that focus on solving real-world problems through a unified I/O interface.

### Tool System

- [x] Model Context Protocol (MCP)
      [Read More](docs/mcp.md)
- [x] Agent with dynamic tool loading
- [x] Browser for web interaction
- [x] Memory integration
  - [x] QDrant vector store
  - [x] Neo4j graph database
- [x] Docker integration
- [x] Github integration
- [x] Azure DevOps integration
- [x] File Editor
- [x] Environment tool for terminal interaction

### Workflow System

- [x] Pipeline Architecture
  - [x] Composable components
  - [x] Bidirectional data flow
- [x] Feedback Loops
      [Read More](feedback.md)
- [ ] Graph-based Workflows
- [x] Stream Processing
      [Read More](stream/README.md)
- [x] Data Conversion

[Read More](workflow/README.md)

### AI Provider Integration

- [x] OpenAI with full capabilities
  - [x] Streaming responses
  - [x] Tool calling
  - [x] Structured outputs
  - [x] Embeddings
- [x] OpenAI-Compatible APIs
- [x] Anthropic (Claude)
  - [x] Streaming responses
  - [x] Tool calling
  - [ ] Structured outputs
- [x] Google
  - [ ] Streaming
  - [ ] Tool calling
  - [ ] Structured outputs
- [x] Cohere
  - [ ] Streaming
  - [ ] Tool calling
  - [ ] Structured outputs
- [x] Ollama (Local Models)
  - [ ] Streaming
  - [ ] Tool calling
  - [ ] Structured outputs

### Security & Data

- [x] Encrypted Payloads
- [x] Artifact System
- [x] Metadata Management
- [x] Cryptographic Signatures

## 🚀 Quick Start

### Prerequisites

- Go 1.21 or higher
- Docker and Docker Compose
- API keys for desired providers

### Installation

```bash
go get github.com/theapemachine/caramba
```

### Basic Usage

Start required services:

```bash
docker compose up
```

> The required services setup is being phased out, given that Caramba now has Docker and Kubernetes built into the framework, including everything that is needed to set up the services.

Run an example:

```bash
go run main.go examples pipeline
```

## 📚 Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Core Concepts](docs/core-concepts.md)
- [API Reference](docs/api-reference.md)
- [Examples](docs/examples.md)

## 🛠️ Example: Creating a Simple Agent

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
    workflow.NewFeedback(provider, agent),
    workflow.NewConverter(),
)
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](docs/contributing.md) for details on how to submit pull requests, report issues, and contribute to the project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
