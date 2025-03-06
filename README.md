# 🤖 Caramba

A specialized agent framework in Go.

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

Caramba follows a strict "everything is `io`" filosophy, meaning that all objects implement at least `io.ReadWriteCloser`.

This allows you to easily connect anything to anything, and is a direct result of the difficulties of creating complex workflows.

## ✨ Features

Caramba comes with a wide array of features that mostly focus on solving real-world problems.

- Support for multiple providers
  - [x] OpenAI
  - [x] OpenAI-Compatible
  - [ ] Anthropic
  - [ ] Google
  - [ ] Cohere
  - [ ] Ollama
- Tool calling
  - [ ] Model Context Provider (MCP)
  - [ ] QDrant
  - [ ] Neo4j
  - [ ] Docker
  - [ ] Browser
  - [ ] Github
  - [ ] Azure DevOps
- Structured outputs
  - [x] OpenAI
  - [x] OpenAI-Compatible
  - [ ] Anthropic
  - [ ] Google
  - [ ] Cohere
  - [ ] Ollama
- Streaming
  - [x] OpenAI
  - [x] OpenAI-Compatible
  - [ ] Anthropic
  - [ ] Google
  - [ ] Cohere
  - [ ] Ollama
- Embeddings
  - [x] OpenAI
  - [x] OpenAI-Compatible
  - [ ] Anthropic
  - [ ] Google
  - [ ] Cohere
  - [ ] Ollama
- (long-term) Memory
  - [x] QDrant
  - [x] Neo4j
- Workflows
  - [x] Pipeline
  - [ ] Graph

## Quick Start

Start the required services.

```bash
docker compose up
```

Run an example.

```bash
go run main.go examples pipeline
```
