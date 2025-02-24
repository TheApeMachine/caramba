# Caramba

This is a sophisticated, real-time, multi-agent system, which highly dynamic, autonomous behavior.

## Core Features

- Agents are dynamically created by other Agents, in principle, creating just one "UI" agent should technically be enough to kick off the whole system.
- Agents should be able to iterate. Instead of just performaing a single generation of a response, each response, and any additional context that may follow that response (for example, the result of any tool calls), should be added to a running Message thread, and fed back into the Agent upon each iteration. This will unlock the ability for self-reflection, and other more advanced behavior. This does mean Agents need a way to indicate that they have completed their current task, so they can break the iteration loop (likely using a tool call?).
- A message queue allows Agents to communicate across private messages, topics, and global broadcast, which enables collaboration.
- Agents should have hierarchy, where there are orchestrators, that deal with teamleads, which deal with team members, etc. This will allow the system to break down highly complex and large cognetive workloads into manageable parts. Theoretically, agents from one team should not be able to talk directly to agents from another team, and messages flow down and up according to the hierarchy.
- Orchestration agents, and teamleads should understand various types of collaboration patterns, for example, an orchestrator could have multiple teams compete for the best answer, or teamleads could lead a discussion between the members on the team, etc.
- Practically NOTHING should be hard-coded when it comes to this process. The system should dynamically build itself up, based purely on the strength of large language models, and the prompts of the initial agents that are created. In the most ideal scenario, only the "ui" agent is created manually, which deals directly with the user, receiving the user prompt, and sending it down-stream, creating the first of the orchestrator agents (there should be a tool to allow the creation of agents), and from there the orchestrator creates teams, and so on.
- All prompt templates etc. must be held in the config file, and used via Viper.
- All data transport should use datura Artifacts, but we must NEVER change anything about datura Artifact, besides adding enums for Roles and Scopes.

The implementations should always be as clean and minimal as possible. Always simplify where possible, always find a more elegant solution. Never write large blocks of code, but break things up into small, manageable structures.

## Implementation Details

### 1. Agent Communication Flow

Current Issue:

- Message routing sometimes fails silently (especially for new agents)
- Queue operations aren't thread-safe
- No verification that messages are actually processed

Proposed Solution:

```go
type Queue struct {
    mu      sync.RWMutex
    agents  map[string]*Agent
    topics  map[string][]*Agent
}

func (q *Queue) SendMessage(msg *Message) error {
    q.mu.RLock()
    defer q.mu.RUnlock()

    target := q.agents[msg.To]
    if target == nil {
        errnie.Error(fmt.Errorf("no agent found: %s", msg.To))
        return fmt.Errorf("agent not found: %s", msg.To)
    }

    // Ensure agent is ready to receive messages
    if target.state == AgentStateInitializing {
        return fmt.Errorf("agent not ready: %s", msg.To)
    }

    select {
    case target.msgs <- msg:
        return nil
    default:
        return fmt.Errorf("agent message queue full: %s", msg.To)
    }
}
```

### 2. Agent Creation and Registration

Current State:

- Good feedback via XML-formatted context updates
- Clear logging through errnie
- Proper agent registration in queue

Remaining Issue:

- Need to ensure agent is fully ready before allowing message processing
- Better coordination between agent creation and executor startup

Proposed Solution:

```go
func (tool *AgentTool) Use(agent *Agent, artifact *Artifact) error {
    decrypted, err := utils.DecryptPayload(artifact)
    if err != nil {
        errnie.Error(err)
        agent.AddContext("<error>Failed to decrypt payload</error>")
        return err
    }

    // Create new agent
    newAgent := ai.NewAgent(
        ai.NewIdentity(params.Role),
        []provider.Tool{
            NewMessageTool().Convert(),
            NewAgentTool().Convert(),
        },
    )

    // Add to queue first
    if err := system.NewQueue().AddAgent(newAgent); err != nil {
        agent.AddContext("<error>Failed to register new agent</error>")
        return err
    }

    // Start executor
    executor := environment.NewExecutor(newAgent)
    if err := environment.NewPool().AddExecutor(executor); err != nil {
        agent.AddContext("<error>Failed to start agent executor</error>")
        system.NewQueue().RemoveAgent(newAgent.Identity.ID)
        return err
    }

    // Only add to parent's agents after everything is ready
    agent.Agents[params.Role] = append(agent.Agents[params.Role], newAgent)

    // Existing feedback is good, just add ready state
    agent.AddContext(fmt.Sprintf(
        "<agent>\n\t<id>%s</id>\n\t<name>%s</name>\n\t<role>%s</role>\n\t<status>READY</status>\n</agent>",
        newAgent.Identity.ID,
        newAgent.Identity.Name,
        newAgent.Identity.Role,
    ))

    return nil
}
```

### 3. Message Processing

Current Issue:

- No clear confirmation of message processing
- Iteration state can get stuck

Proposed Solution:

```go
func (executor *Executor) handleMessage(msg *datura.Artifact) *datura.Artifact {
    if msg == nil {
        errnie.Error(fmt.Errorf("received nil message"))
        return nil
    }

    decrypted, err := utils.DecryptPayload(msg)
    if errnie.Error(err) != nil {
        executor.Agent.AddContext("<error>Failed to decrypt message</error>")
        return nil
    }

    // Add processing confirmation to context
    executor.Agent.AddContext("<message>Processing: " + string(decrypted) + "</message>")

    // ... rest of processing ...

    return processed
}
```

### Implementation Priority

1. Message Flow Reliability

   - Thread-safe queue operations
   - Message processing confirmation
   - Better state management for new agents

2. Agent Coordination

   - Ensure proper startup sequence
   - Clear ready states
   - Safe shutdown

3. Iteration Control
   - Better completion detection
   - Clear iteration state changes
   - Prevent stuck states

The focus is on making the communication flow more reliable while preserving the good feedback mechanisms already in place.
