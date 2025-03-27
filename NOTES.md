# NOTES

Keep any important information, goals, insights, etc. by writing them down in this file.

It will be attached to each request so you will have access to it.

## Project Overview

Caramba is a Go-based agent framework with the following key characteristics:

1. Core Philosophy:

   - Everything is treated as `io.ReadWriteCloser`
   - Unified interface for connecting different components
   - Strong focus on agent-based architecture

2. Main Components:

   - `/pkg/ai`: AI-related functionality
   - `/pkg/provider`: Various service providers (OpenAI, Anthropic, etc.)
   - `/pkg/workflow`: Pipeline and workflow management
   - `/pkg/tools`: Framework tools
   - `/pkg/memory`: Memory management systems
   - `/pkg/core`: Core framework functionality
   - `/pkg/system`: System-level operations
   - `/pkg/kube`: Kubernetes integration
   - `/pkg/stream`: Stream processing capabilities
   - `/pkg/service`: Service management
   - `/pkg/process`: Process handling
   - `/pkg/fs`: File system operations
   - `/pkg/errnie`: Error handling

3. Key Features:

   - Model Context Protocol (MCP) server capabilities
   - Multiple AI provider integrations (OpenAI, Claude, Google, Cohere, Ollama)
   - Docker and Kubernetes integration
   - Browser automation
   - Memory integration (QDrant vector store, Neo4j graph database)
   - Pipeline architecture with bidirectional data flow
   - Stream processing
   - Tool system with dynamic loading
   - DevOps integrations (Github, Azure DevOps)

4. Technical Stack:

   - Go 1.24.0
   - Uses modern Go toolchain (go1.24.1)
   - Extensive dependency list including containerd, docker, kubernetes clients
   - Support for various AI/ML providers and tools

5. Project Status:
   - Active development (based on recent commits)
   - Well-structured modular architecture
   - Comprehensive test coverage (based on CI badges)
   - Strong focus on security and containerization
