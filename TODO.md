# The Road to MCP

This document describes what needs to be done to move more in the direction of using Model Context Protocol as the main instrument for orchestration of tools, resources, and processes.

For clarity, let's define what we mean by those terms.

## Tools

Anything that enables a Large Language Model to do anything beyond generating text-based responses that have no side-effects.

This could be interfacing with external APIs, running code in Docker containers, sending messages to other agents, breaking out of iteration loops, etc.

## Resources

Any data an agent operates on using tools, or needed as additional context for the LLM to generate a more effective and relevant response.

## Processes

A guided (controlled) way to get a more predictable response from a LLM, potentially with incorporating multiple generation cycles, and/or tool calls.

This is related, but not entirely the same as how we previously used structured outputs.

---

## Priorities

At all times, we need to keep in sharp focus that Developer Experience (DX) is key in this agent framework, because we want to end up with a situation where we can freely and easily experiment with AI Agents, and not fight complexity, or get lost in boilerplate.

The public facing API should be as clean as possible, and we should always keep reasoning about how we can make things even simpler, or remove even more boilerplate.

Let's look at a couple of idyllic examples, in the form of what is also our primary goal to achieve for a first version of the framework: AI code generation, with long-horizon goals, planning, verification, and self-optimization.

```go
package examples

/*
Development is an example AI agent workflow.
*/
type Development struct {
    wf *workflow.Graph
}

/*
NewDevelopment sets up the runtime environment.
*/
func NewDevelopment() *Development {
    return &Development{
        wf: workflow.NewGraph(
            workflow.WithNodes(
                ai.NewAgent(
                    ai.WithIdentity("agent1"),
                    ai.WithRole("manager"),
                    ai.WithContext(
                        ai.NewContext(
                            ai.WithSystemPrompt(tweaker.SystemPrompt("default"))
                        )
                    ),
                    ai.WithTools(
                        tools.Azure, tools.Browser,
                    ),
                    ai.WithMemory(
                        memory.Conversation, memory.LongTerm, memory.Global,
                    ),
                    ai.WithOptimizer(
                        "params", "finetuner",
                    ),
                ),
                ai.NewAgent(
                    ai.WithIdentity("agent2"),
                    ai.WithRole("developer"),
                    ai.WithContext(
                        ai.NewContext(
                            ai.WithSystemPrompt(tweaker.SystemPrompt("default"))
                        )
                    ),
                    ai.WithTools(
                        tools.Environment, tools.Browser,
                    ),
                    ai.WithMemory(
                        memory.Conversation, memory.LongTerm, memory.Global,
                    )
                    ai.WithOptimizer(
                        "params", "finetuner",
                    ),
                ),
                ai.NewAgent(
                    ai.WithIdentity("agent3"),
                    ai.WithRole("reviewer"),
                    ai.WithContext(
                        ai.NewContext(
                            ai.WithSystemPrompt(tweaker.SystemPrompt("default"))
                        )
                    ),
                    ai.WithTools(
                        tools.Environment, tools.Browser,
                    ),
                    ai.WithMemory(
                        memory.Conversation, memory.LongTerm, memory.Global,
                    )
                    ai.WithOptimizer(
                        "params", "finetuner",
                    ),
                ),
            ),
            workflow.WithEdges(
                workflow.WithBiDirectional("agent1", "agent3"),
                workflow.WithForward("agent1", "agent2"),
                workflow.WithBackward("agent2", "agent3")
            ),
        )
    }
}
```

In the example above, we should assume that agents all run in their own execution context, using a goroutine to make them concurrent, and that they receive messages from a hub of some kind, which can be private (agent to agent, or system to agent), topic based (subscription), or global (broadcast).

Prompt and resource selection should be done based of at least a combination of the role of the agent, and the current tool it has selected to use. This is likely going to expand later and become slightly more complex still.

Agents should iterate on tasks, until they have completed and reached their current (sub) goal.

(Self) optimization makes use of "system-level" AI agents, which can activate at certain stages (for instance when another agent has completed a sub-goal), and receive the full context of the agent at the time of sub-goal completion. This allows the optimizer agent to evaluate the historical slice of that agent's work on the sub-goal, and decide to either tweak the model params of the agent (temperature, for example), or even tweak the agent's system prompt, for even more control over its behavior.
Secondary to that, it can also extract training data from the agent's context and store it in a format that can be used for fune-tuning (provided the agent produced high-quality responses along the way), enabling us to periodically and automatically kick of fine-tuning runs.

## Current Codebase Analysis & MCP Integration Path

### Current Architecture Analysis

After deeper investigation, it appears the codebase is already more MCP-ready than initially thought:

1. **Existing MCP Integration**

   - All tools already implement MCP compatibility via `Schema.ToMCP()`
   - Both stdio and SSE MCP servers are implemented
   - Tools are already bridged between I/O and MCP systems
   - The service layer handles MCP protocol translation

2. **Current Implementation**

   ```go
   // Tools are registered with MCP server
   service.stdio.AddTool(
       service.tools["toolname"].Schema.ToMCP(),
       func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
           return service.runTool(service.tools["toolname"], &req, datura.ArtifactRole...)
       },
   )

   // Bridge between I/O and MCP already exists
   func (service *MCP) runTool(tool io.ReadWriteCloser, req *mcp.CallToolRequest, role datura.ArtifactRole) (*mcp.CallToolResult, error)
   ```

### Revised Integration Strategy

Given this discovery, we should focus on:

1. **Process Management**

   - Build on existing MCP integration
   - Add multi-step process capabilities
   - Implement process templates
   - Add state management

2. **Resource Control**

   - Enhance existing tool schemas
   - Add resource binding
   - Implement access control
   - Add resource templates

3. **Role-Based Access**

   - Define role capabilities
   - Map tools to roles
   - Create process permissions
   - Build role templates

4. **Implementation Phases**

   Phase 1: Process Engine

   - Create process definition format
   - Implement process executor
   - Add state management
   - Build process monitoring

   Phase 2: Resource Management

   - Enhance resource capabilities
   - Add resource templates
   - Implement resource binding
   - Build access control

   Phase 3: Role System

   - Define role structure
   - Create role templates
   - Implement permissions
   - Add role inheritance

   Phase 4: Integration

   - Connect processes to roles
   - Bind resources to steps
   - Add monitoring
   - Implement optimization

### Next Steps

1. Design the process definition format
2. Create the process execution engine
3. Implement basic state management
4. Build example multi-step processes

This revised approach leverages the existing MCP integration while focusing on building the higher-level capabilities needed for structured agent interactions.

## Unified Streaming Architecture

The codebase currently implements io.ReadWriteCloser across many components (tools, providers, workflows, etc.) with similar patterns. We can unify this into a single, powerful abstraction:

### Core Architecture

1. **Universal Streamer Base**

```go
// Base streaming component for all streamable types
type Streamer struct {
    buffer *stream.Buffer
}

// Interface for any component that can be streamed
type StreamComponent interface {
    Handle(artifact *datura.Artifact) error
}

// Creates a new streaming component with options
func NewStreamer(component StreamComponent, opts ...StreamerOption) *Streamer
```

1. **Unified Transport Layer**

- Uses datura.Artifact as a universal envelope
- Maintains type safety through Cap'n Proto
- Supports any Cap'n Proto message as payload
- Uses metadata to describe transported types

1. **Component Implementation**

```go
// Example component
type MyComponent struct {
    *Streamer
    // Component-specific fields
}

// Only need to implement component-specific logic
func (c *MyComponent) Handle(artifact *datura.Artifact) error {
    // Component logic here
    return nil
}
```

### Benefits

1. **Reduced Boilerplate**

   - No repeated io.ReadWriteCloser implementations
   - Components only implement their specific logic
   - Common streaming behavior unified in one place

2. **Enhanced Capabilities**

   - Middleware support across all components
   - Consistent error handling
   - Unified metrics and logging
   - Easy to add new features to all components

3. **Type Safety**

   - Maintains Cap'n Proto type safety
   - Universal datura.Artifact envelope
   - Type-safe payload transportation

4. **Flexibility**
   - Support for any Cap'n Proto message type
   - Easy to add new component types
   - Middleware can be added per-component

### Implementation Example

```go
// Component creation with options
func NewComponent() *MyComponent {
    comp := &MyComponent{
        // Initialize component-specific fields
    }
    comp.Streamer = NewStreamer(
        comp,
        WithMetrics(),
        WithLogging(),
        WithMiddleware(customMiddleware),
    )
    return comp
}

// Usage remains simple
component := NewComponent()
io.Copy(destination, component)
```

### Streaming Next Steps

1. **Migration Path**

   - Create the base Streamer implementation
   - Add middleware support
   - Create common middleware (logging, metrics, etc.)
   - Gradually migrate existing components

2. **Enhancement Opportunities**

   - Add tracing support
   - Implement backpressure handling
   - Add circuit breakers
   - Support for streaming pools

3. **Documentation**
   - Document migration process
   - Create examples for common use cases
   - Document middleware creation
   - Create component templates

This unified approach will significantly reduce code duplication while making the system more maintainable and easier to extend.

### Generator Pattern for Multi-Directional Streaming

To address the complexity of multi-directional streaming, we can implement components as Generators that run concurrently and communicate through a central message hub:

1. **Generator Base**

```go
// Base generator type for concurrent streaming components
type Generator struct {
    *Streamer
    ctx    context.Context
    cancel context.CancelFunc
    hub    *MessageHub
}

// Interface for generator components
type GeneratorComponent interface {
    StreamComponent
    Generate()
}

// Creates a new generator with hub connection
func NewGenerator(component GeneratorComponent, hub *MessageHub) *Generator {
    ctx, cancel := context.WithCancel(context.Background())
    return &Generator{
        ctx:    ctx,
        cancel: cancel,
        hub:    hub,
    }
}
```

1. **Message Hub**

```go
// Central message routing
type MessageHub struct {
    routes   map[string]chan *datura.Artifact
    mu       sync.RWMutex
}

func (h *MessageHub) Subscribe(topic string) chan *datura.Artifact
func (h *MessageHub) Publish(topic string, msg *datura.Artifact)
func (h *MessageHub) Connect(from, to string) // Create route between generators
```

1. **Component Implementation**

```go
// Example generator component
type MyGenerator struct {
    *Generator
}

func NewMyGenerator(hub *MessageHub) *MyGenerator {
    gen := &MyGenerator{}
    gen.Generator = NewGenerator(gen, hub)
    return gen
}

func (g *MyGenerator) Generate() {
    go func() {
        for {
            select {
            case <-g.ctx.Done():
                return
            case msg := <-g.hub.Subscribe("my-topic"):
                // Process message
                result := g.process(msg)
                g.hub.Publish("output-topic", result)
            }
        }
    }()
}

func (g *MyGenerator) Handle(artifact *datura.Artifact) error {
    // Optional direct handling (can delegate to Generate())
    return nil
}
```

### Generator Benefits

1. **Concurrent by Default**

   - All components run in their own goroutines
   - Non-blocking message handling
   - Natural backpressure through channels

2. **Flexible Routing**

   - Dynamic message routing through hub
   - Support for pub/sub patterns
   - Easy to add new communication paths

3. **Clean Shutdown**

   - Controlled shutdown through context
   - Resource cleanup on cancellation
   - Graceful component termination

4. **Workflow Integration**

```go
// Example workflow setup
workflow := NewWorkflow()
hub := NewMessageHub()

// Create generators
gen1 := NewMyGenerator(hub)
gen2 := NewOtherGenerator(hub)

// Connect generators based on workflow
hub.Connect("gen1-output", "gen2-input")

// Start generators
gen1.Generate()
gen2.Generate()
```

### Implementation Strategy

1. **Core Components**

   - Implement Generator base type
   - Create MessageHub with routing
   - Add workflow-based connection setup

2. **Message Patterns**

   - Point-to-point communication
   - Pub/sub topics
   - Broadcast capabilities
   - Message filtering

3. **Workflow Integration**

   - Define workflow-based routing
   - Support dynamic reconfiguration
   - Add monitoring and control

4. **Error Handling**
   - Retry mechanisms
   - Dead letter channels
   - Error propagation
   - Circuit breaking

This generator pattern complements the unified streaming architecture by providing a clean solution for concurrent, multi-directional communication while maintaining the benefits of the Streamer base.

## Distributed Service Architecture

The framework needs to support running tools as distributed services across multiple servers, while maintaining a simple developer experience. Here's how this will work:

### Service-Based Tool Architecture

1. **Tool as a Service**

   ```bash
   # Start a tool as an MCP service
   caramba serve tool [tool-name] --transport [stdio|sse] --port 8080
   ```

   - Each tool can run as a standalone MCP service
   - Support both STDIO (local) and SSE (remote) transports
   - Configurable through standard CLI flags
   - Auto-registration with service discovery

2. **Transport Modes**

   a. **STDIO Mode**

   - Local subprocess communication
   - Direct integration with desktop apps
   - High performance for local tools
   - Example: `caramba serve tool editor --transport stdio`

   b. **SSE Mode**

   - Network-based communication
   - Multiple client support
   - Cross-machine tool access
   - Example: `caramba serve tool github --transport sse --port 8080`

3. **Tool Discovery**

   Phase 1 (Simple):

   - Static configuration file for known tools
   - Manual registration of remote tools
   - Basic health checking

   ```yaml
   # tools-registry.yaml
   tools:
     github:
       url: http://github-tools.internal:8080
       transport: sse
       health: /health
     editor:
       path: /usr/local/bin/caramba-editor
       transport: stdio
   ```

   Phase 2 (Advanced):

   - Automatic service discovery
   - Dynamic tool registration
   - Health monitoring
   - Load balancing

### Developer Experience

1. **Code-Level Transparency**

   ```go
   // Developer doesn't need to know if tool is local or remote
   agent := ai.NewAgent(
       ai.WithTools(
           tools.Github,  // Could be remote
           tools.Editor,  // Could be local
       ),
   )
   ```

2. **Configuration Over Code**

   - Tool location/transport defined in config
   - No code changes needed for different deployments
   - Runtime discovery and connection management

3. **Local Development**

   ```bash
   # Start all tools locally for development
   caramba dev

   # Start specific tools
   caramba dev --tools github,editor
   ```

### Implementation Plan

1. **CLI Enhancement**

   - Add `serve` command for tool services
   - Support transport configuration
   - Add service discovery flags
   - Implement health checking

2. **Service Discovery**

   - Create simple registry interface
   - Implement file-based registry
   - Add health check endpoints
   - Prepare for advanced discovery

3. **Tool Wrapper**

   - Create transport-agnostic tool wrapper
   - Implement connection management
   - Add retry and failover logic
   - Support tool versioning

4. **Development Mode**
   - Create development environment
   - Auto-start required tools
   - Support mixed local/remote setup
   - Add debugging capabilities

This architecture allows us to:

- Scale tools independently
- Mix local and remote tools seamlessly
- Maintain simple developer experience
- Prepare for more advanced service discovery

Next Steps:

1. Implement basic `serve` command
2. Create simple file-based registry
3. Update tool initialization to support remote connections
4. Add development mode support
