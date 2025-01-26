package ai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/ai/drknow"
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
	Identity      *drknow.Identity `json:"identity" jsonschema:"title=Identity,description=The identity of the agent,required"`
	Context       *drknow.Context  `json:"context" jsonschema:"title=Context,description=The context of the agent,required"`
	MaxIterations int              `json:"max_iterations" jsonschema:"title=Max Iterations,description=The maximum number of iterations to perform,required"`
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
	out := make(chan provider.Event)

	go func() {
		defer close(out)

		// Add timeout for agent generation
		genCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
		defer cancel()

		shouldBreak := false
		cycle := 0

		msg.Content = utils.JoinWith(
			"[USER PROMPT]",
			msg.Content,
			"[/USER PROMPT]",
		)

		agent.Context.AddMessage(msg)

		for !shouldBreak {
			agent.state = AgentStateGenerating

			cycle++
			if cycle >= agent.MaxIterations {
				shouldBreak = true
			}

			// // Run recall task and add results to context
			// agent.task("recaller", "recall", out)

			compiled := agent.Context.Compile(cycle, agent.MaxIterations)
			errnie.Log("compiled: %v", compiled)

			for event := range agent.accumulator.Generate(
				genCtx,
				agent.provider.Generate(
					genCtx,
					compiled,
				),
			) {
				out <- event
			}

			response := agent.accumulator.String()

			agent.Context.AddMessage(
				provider.NewMessage(
					provider.RoleAssistant,
					response,
				),
			)

			if strings.Contains(
				strings.ToLower(response),
				"<break>",
			) {
				shouldBreak = true
			}

			agent.Context.AddMessage(
				provider.NewMessage(
					provider.RoleAssistant,
					fmt.Sprintf("<<< END iteration %d of %d", cycle+1, agent.MaxIterations),
				),
			)
		}

		// Run optimization task with full context but don't add results
		// agent.task("optimizer", "optimize", out)
	}()

	return out
}

func (agent *Agent) task(system string, task string, out chan<- provider.Event) {
	accumulator := stream.NewAccumulator()
	v := viper.GetViper()
	systemPrompt := v.GetString("prompts.templates.systems." + system)

	ctx := drknow.QuickContext(
		systemPrompt,
		"codeswitch",
		"noexplain",
		"silentfail",
		"scratchpad",
	)

	taskPrompt := v.GetString("prompts.templates.tasks." + task)

	// Add task prompt with context
	ctx.AddMessage(
		provider.NewMessage(
			provider.RoleUser,
			taskPrompt,
		),
	)

	errnie.Log("THREAD for task:\n%v", task)

	// Generate response and process through accumulator once
	for event := range accumulator.Generate(
		ctx.Identity.Ctx,
		agent.provider.Generate(
			ctx.Identity.Ctx,
			ctx.Compile(1, 1),
		),
	) {
		// Forward the event
		out <- event
	}

	// Process the task response using the interpreter
	interpreter := NewInterpreter(agent.Context, accumulator)
	interpreter.Interpret().Execute()
}

func (agent *Agent) Validate() bool {
	if agent.Context == nil {
		errnie.Error(errors.New("context is nil"))
		return false
	}

	if agent.Context.Identity == nil {
		errnie.Error(errors.New("identity is nil"))
		return false
	}

	if err := agent.Context.Identity.Validate(); err != nil {
		errnie.Error(err)
		return false
	}

	if agent.provider == nil {
		errnie.Error(errors.New("provider is nil"))
		return false
	}

	return true
}
