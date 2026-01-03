# A2A Architecture

This document describes the A2A (Agent-to-Agent) architecture for the Caramba platform.

## Overview

The A2A architecture converts every persona into an A2A-served expert agent running in Docker. Users interact with a Root agent via a Textual TUI, and the Root agent delegates to expert personas as needed.

## Components

### Persona A2A Servers

Each persona YAML (except legacy provider personas) runs as an A2A server in a Docker container:

- **Service name**: Persona name (lowercase, hyphens for underscores)
- **Port**: 8001 (internal, not exposed to host)
- **Agent card**: `http://<service_name>:8001/.well-known/agent-card.json`

### Root Agent

The Root agent orchestrates expert personas:

- **Persona**: `config/personas/root.yml`
- **Service**: `root-agent` (exposed on host port 9000)
- **Sub-agents**: All expert personas configured in `root.yml`

### RandomModel Provider

Every persona uses a RandomModel provider that selects one of three models per request:

- `openai/gpt-5.2`
- `anthropic/claude-opus-4-5-20251101`
- `google/gemini-3-pro-preview`

This ensures persona behavior is consistent while reducing model bias.

### Textual TUI

The Textual TUI (`cli/tui.py`) provides:

- **Chat viewport**: Streaming responses from Root agent
- **Experts sidebar**: Shows which experts are being consulted
- **Tools sidebar**: Shows tool calls made by agents

## Usage

### Starting the System

1. Start all services:
   ```bash
   docker compose up
   ```

2. Run the TUI:
   ```bash
   python -m cli.tui
   ```

   Or set `ROOT_AGENT_URL` to customize the root agent URL:
   ```bash
   ROOT_AGENT_URL=http://localhost:9000 python -m cli.tui
   ```

### Architecture Flow

```
User → Textual TUI → Root Agent (A2A) → Expert Agents (A2A) → MCP Tools
```

1. User types message in TUI
2. TUI sends to Root agent (port 9000)
3. Root agent decides which experts to consult
4. Root agent calls experts via A2A (using docker-compose hostnames)
5. Experts may call MCP tools (filesystem, codegraph, etc.)
6. Responses stream back through Root to TUI

## Configuration

### Persona Configuration

Personas can declare:

- `sub_agents`: List of sub-agent names (for delegation)
- `model`: Model ID or "random" (defaults to RandomModel)
- `tools` or `mcp_servers`: MCP tool server names

### Docker Compose

Each persona gets a service in `docker-compose.yml`:

```yaml
persona-name:
  build:
    context: .
    dockerfile: docker/Dockerfile.persona-a2a
  environment:
    - PERSONA_FILE=config/personas/persona_name.yml
    - PORT=8001
  volumes:
    - ./:/app
```

## Future Extensions

- **Verifier/Critic agents**: Can be added as additional personas
- **Sub-agent chains**: Any persona can have sub-agents (not just Root)
- **Custom model pools**: Per-persona model selection policies
