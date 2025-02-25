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

## ✨ Features

Caramba comes with a wide array of features that mostly focus on solving real-world problems.

### 🧠 Multi-Provider Intelligence

With all the major providers implemented, and the ability to define a custom API endpoint for the OpenAI provider, the framework should be compatible with most providers.

The is also a "synthetic" provider called the `BalancedProvider` which automatically load-balances generation calls across all currently active providers, taking care of error handling, cooldown periods, etc.

### Iteration

Agents are enabled to iterate, meaning that their responses, and any side-effects to their responses, such as tool call results, are collected and added to their current context, which the loops around and the agent is re-prompted with the updated context. This allows an agent to iterate over their output, enabling things like self-reflection, etc.

An agent is instructed to indicate once they are happy with the results, and they will break out of this iteration loop.

### Communication

Agents are able to send messages to each other, to a topic, or as a broadcast. Other agents can subscribe to topics, or create new ones.

This allows for collaboration.

### Memory

Agents in the Caramba framework can have two distinct types of memory, vector-based, and graph-based.

- **QDrant**: vector-based memory store
  - Agents can have their own personal long-term memory
  - Agents can be connected to a global long-term memory
- **Neo4j**: grah-based memory store
  - Agents can have their own personal long-term memory
  - Agents can be connected to a global long-term memory

These memory stores are abstracted away behind a unified memory layer, which automatically extracts memories from an agent's current context, if the content is interesting enough to be stored in long-term memory. The idea would be to use the help of an LLM to determine which parts of the current context to store as
memories, and similarly, to have an LLM sit at the front of an initial Agent call, and use the incoming prompt to generate queries to call the memory layer with.

```txt
-[user prompt]-> -[memory recall/queries]-> -[inject found memories into context]-> -[prompt agent]-> -[handle tool calls]-> -[extract memories]->
```

Before an agent starts its first response, the unified memory layer is automatically queried for relevant long-term memories, and anything that was found is injected into the current context, before the agent is first prompted, so they will have those memories available to them.

### Tools

A good selection of tools is integrated out of the box, but of course the Tool interface allows anyone to easily add their own custom toolsets.

- **Agent**: potentially one of the most powerful tools
  - Allows an agent to create other agents on-the-fly
- **Docker**: a fully featured Debian (or customized) execution environment
  - Using the Docker API, agents are able to make use of a full linux shell, running in a Docker container
- **Browser**: a fully featured (headless) browser instance
  - Allows an agent not only to browse the web, but also perform complex tasks, using the developer tools
  - Features include:
    - Navigate to websites and retrieve HTML content
    - Take screenshots of web pages
    - Generate PDF files from web pages
    - Extract specific content using CSS selectors
    - Execute custom JavaScript on web pages
- **Github**: integration with the Github API + WebHooks
  - Allows an agent to act as a fully integrated developer
  - Allows an agent to perform code-reviews
  - Enables an agent to search code
  - Features include:
    - Search for code across repositories with language and path filters
    - Get repository information and metrics
    - List and create issues with full metadata support
    - View, create, and comment on pull requests
    - Submit code reviews (approve, comment, or request changes)
    - Retrieve file content from repositories
    - Create and manage comments on issues and pull requests
- **Azure DevOps**: integration with the Azure DevOps API + WebHooks
  - Mostly focused on the Boards, for project management
  - Features include:
    - Create, read, update, and query work items (Epics, Features, User Stories, Bugs, Tasks)
    - Manage work item comments and attachments
    - Get project details and list available projects
    - List and manage teams within projects
    - Access sprint information and track progress
    - Execute custom WIQL queries for advanced work item filtering
- **Slack**: integration with the Slack API + WebHooks
  - Allows an agent to have a two-way communication channel with a team
  - Allows an agent to respond to conversations and perform background tasks
  - Enables an agent to search Slack history

### Self-Optimiation

A Caramba-based Agent "system" should be able to self-optimize.

- **Verification**: special LLM-based verifier units
  - Verify and assess the performance of an agent, and are capable of adjusting model parameters, such a temperature, and other values, as well as tweak the agent's system prompt
  - Extract any high quality prompt/response examples, and store them in a format ready for periodic fine-tuning of models
