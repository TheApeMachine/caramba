package ai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
AgentState represents the current operational state of an Agent during its lifecycle.
It tracks whether the agent is idle, generating responses, calling tools, or completed its task.
*/
type AgentState uint

const (
	AgentStateIdle AgentState = iota
	AgentStateGenerating
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
	Identity      *Identity `json:"identity" jsonschema:"title=Identity,description=The identity of the agent,required"`
	Context       *Context  `json:"context" jsonschema:"title=Context,description=The context of the agent,required"`
	MaxIterations int       `json:"max_iterations" jsonschema:"title=Max Iterations,description=The maximum number of iterations to perform,required"`
	provider      provider.Provider
	accumulator   *stream.Accumulator
	state         AgentState
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
func NewAgent(ctx *Context, role string, maxIterations int) *Agent {
	return &Agent{
		Context:       ctx,
		provider:      provider.NewBalancedProvider(),
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
AddTools appends the provided tools to the agent's available toolset,
expanding its capabilities for task execution.

Parameters:
  - tools: Variable number of tools to add to the agent
*/
func (agent *Agent) AddTools(tools ...provider.Tool) {
	agent.Identity.Params.Tools = append(agent.Identity.Params.Tools, tools...)
}

/*
AddProcess activates structured outputs for the agent by setting a process
that defines a specific JSON schema for response formatting.

Parameters:
  - process: The process definition containing the output schema
*/
func (agent *Agent) AddProcess(process provider.Process) {
	agent.Identity.Params.Process = process
}

/*
RemoveProcess deactivates structured outputs for the agent,
reverting it back to generating freeform text responses.
*/
func (agent *Agent) RemoveProcess() {
	agent.Identity.Params.Process = nil
}

/*
GetRole returns the role designation assigned to this agent,
as defined in its identity.
*/
func (agent *Agent) GetRole() string {
	return agent.Identity.Role
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
func (agent *Agent) Generate(ctx context.Context, msg *provider.Message) <-chan provider.Event {
	errnie.Info("generating response for %s (%s)", agent.Context.identity.Name, msg.Role)
	out := make(chan provider.Event)

	go func() {
		defer close(out)

		if !agent.Validate() {
			errEvent := provider.NewEventData()
			errEvent.EventType = provider.EventError
			errEvent.Error = fmt.Errorf("agent validation failed")
			errEvent.Name = "agent_validation_error"
			out <- errEvent
			return
		}

		shouldBreak := false
		cycle := 0

		for !shouldBreak {
			errnie.Info("cycle %d", cycle)
			agent.state = AgentStateGenerating

			cycle++
			if cycle > agent.MaxIterations {
				shouldBreak = true
			}

			// Add timeout for agent generation
			genCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()

			for event := range agent.accumulator.Generate(
				genCtx,
				agent.provider.Generate(
					genCtx,
					agent.Context.Compile(msg, cycle, agent.MaxIterations),
				),
			) {
				out <- event
			}

			if strings.Contains(
				strings.ToLower(agent.accumulator.String()),
				"<break>",
			) {
				shouldBreak = true
			}
		}

		errnie.Info("optimization started")
		agent.Optimize(ctx, out)
		errnie.Info("optimization complete")
	}()

	return out
}

func (agent *Agent) Optimize(ctx context.Context, out chan<- provider.Event) {
	accumulator := stream.NewAccumulator()
	v := viper.GetViper()
	system := v.GetString("prompts.templates.systems.optimizer")

	if system == "" {
		errnie.Error(errors.New("optimizer system is empty"))
		return
	}

	thread := provider.NewThread().AddMessage(
		provider.NewMessage(
			provider.RoleSystem,
			system,
		),
	).AddMessage(
		provider.NewMessage(
			provider.RoleUser,
			utils.ReplaceWith(
				v.GetString("prompts.templates.tasks.optimize"),
				[][]string{
					{"<{context}>", agent.Context.String()},
				},
			),
		),
	)

	errnie.Log("THREAD\n\n%v", thread)

	for event := range accumulator.Generate(
		ctx,
		agent.provider.Generate(
			ctx,
			&provider.LLMGenerationParams{
				Thread: thread,
			},
		),
	) {
		// Forward the event to the output channel
		out <- event
	}

	// Process the optimization response using the interpreter
	interpreter := NewInterpreter(accumulator.String())
	interpreter.Interpret().Execute()
}

func (agent *Agent) Validate() bool {
	if agent.Context == nil {
		errnie.Error(errors.New("context is nil"))
		return false
	}

	if agent.Context.identity == nil {
		errnie.Error(errors.New("identity is nil"))
		return false
	}

	if err := agent.Context.identity.validate(); err != nil {
		errnie.Error(err)
		return false
	}

	if agent.Context.identity.System == "" {
		errnie.Error(errors.New("system is empty"))
		return false
	}

	if agent.provider == nil {
		errnie.Error(errors.New("provider is nil"))
		return false
	}

	return true
}
