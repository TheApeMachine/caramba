package ai

import (
	"context"
	"strings"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/ai/tasks"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/utils"
)

/*
AgentState represents the current operational state of an Agent during its lifecycle.
It tracks whether the agent is idle, generating responses, calling tools, or completed its task.
*/
type AgentState uint

const (
	AgentStateIdle AgentState = iota
	AgentStateGenerating
	AgentStateTerminal
	AgentStateToolCalling
	AgentStateIterating
	AgentStateDone
)

/*
Agent wraps the requirements and functionality to turn a prompt and
response sequence into behavior. It enhances the default functionality
of a large language model by adding optional structured responses and
tool calling capabilities.

The Agent maintains its own identity, context, and state while coordinating
with providers to generate responses and execute tools.
*/
type Agent struct {
	Identity      *drknow.Identity `json:"identity" jsonschema:"title=Identity,description=The identity of the agent,required"`
	Context       *drknow.Context  `json:"context" jsonschema:"title=Context,description=The context of the agent,required"`
	MaxIterations int              `json:"max_iterations" jsonschema:"title=Max Iterations,description=The maximum number of iterations to perform,required"`
	provider      provider.Provider
	accumulator   *stream.Accumulator
	state         AgentState
	bridge        tasks.Bridge
}

/*
NewAgent creates a new Agent instance with the specified role and maximum iterations.
It initializes the agent with a new identity, balanced provider, and accumulator,
setting its initial state to idle.

Parameters:
  - ctx: The context for operations
  - role: The role designation for the AI agent
  - maxIterations: The maximum number of response generation iterations
*/
func NewAgent(ctx *drknow.Context, prvdr provider.Provider, role string, maxIterations int) *Agent {
	return &Agent{
		Context:       ctx,
		provider:      prvdr,
		MaxIterations: maxIterations,
		accumulator:   stream.NewAccumulator(),
		state:         AgentStateIdle,
	}
}

/*
GenerateSchema implements the Tool interface for Agent, allowing agents to create
new agents for task delegation when given access to the Agent tool.
It returns a JSON schema representation of the Agent type.
*/
func (agent *Agent) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Agent]()
}

/*
Generate calls the underlying provider to have a Large Language Model
generate text for the agent. It compiles the context and streams the
response through an accumulator.

Parameters:
  - ctx: The context for the generation operation
  - msg: The message to generate a response for

Returns:
  - A channel of provider.Event containing the generated response
*/
func (agent *Agent) Generate(ctx context.Context, msg *provider.Message) <-chan *provider.Event {
	out := make(chan *provider.Event)

	go func() {
		defer close(out)

		shouldBreak := false
		cycle := 0

		// Add the user prompt, and wrap it in some tags
		msg.Content = utils.QuickWrap(
			"USER PROMPT",
			msg.Content,
			1,
		)

		agent.Context.AddMessage(msg)

		for !shouldBreak {
			cycle++

			for event := range agent.accumulator.Generate(
				ctx,
				agent.provider.Generate(
					ctx,
					agent.Context.Compile(cycle, agent.MaxIterations),
				),
			) {
				out <- event
			}

			response := agent.accumulator.String()
			agent.accumulator.Clear()

			// Add a newline if the response doesn't end with one
			if !strings.HasSuffix(response, "\n") {
				response += "\n"
			}

			agent.Context.AddMessage(
				provider.NewMessage(
					provider.RoleAssistant,
					response,
				),
			)

			// Process commands using the interpreter
			interpreter := NewInterpreter(agent.Context)
			interpreter, agent.state = interpreter.Interpret()
			agent.bridge = interpreter.Execute()

			// Check for break command
			if strings.Contains(strings.ToLower(response), "<break") {
				shouldBreak = true
			}
		}
	}()

	return out
}

func (agent *Agent) Validate() bool {
	if agent.Context == nil {
		log.Error("Context is nil")
		return false
	}

	if agent.Context.Identity == nil {
		log.Error("Identity is nil")
		return false
	}

	if err := agent.Context.Identity.Validate(); err != nil {
		log.Error("Identity is invalid", "error", err)
		return false
	}

	if agent.provider == nil {
		log.Error("Provider is nil")
		return false
	}

	return true
}
