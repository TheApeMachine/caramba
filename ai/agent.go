package ai

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/log"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
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
	AgentStateTerminal
	AgentStateDelegating
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
	Name          string           `json:"name" jsonschema:"title=Name,description=The name of the agent,required"`
	Role          string           `json:"role" jsonschema:"title=Role,description=The role of the agent,required"`
	Identity      *drknow.Identity `json:"identity" jsonschema:"title=Identity,description=The identity of the agent,required"`
	Context       *drknow.Context  `json:"context" jsonschema:"title=Context,description=The context of the agent,required"`
	MaxIterations int              `json:"max_iterations" jsonschema:"title=Max Iterations,description=The maximum number of iterations to perform,required"`
	provider      provider.Provider
	accumulator   *stream.Accumulator
	state         AgentState
	container     *tools.Container
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
func NewAgent(
	ctx *drknow.Context,
	prvdr provider.Provider,
	role string,
	maxIterations int,
) *Agent {
	return &Agent{
		Name:          ctx.Identity.Name,
		Role:          role,
		Identity:      ctx.Identity,
		Context:       ctx,
		MaxIterations: maxIterations,
		provider:      prvdr,
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
	log.Debug("generating response", "role", msg.Role, "content", msg.Content)
	out := make(chan *provider.Event)

	go func() {
		defer close(out)

		cycle := 0

		agent.Context.AddMessage(msg)

		for agent.state != AgentStateDone {
			log.Debug("next cycle", "cycle", cycle)

			cycle++
			params := agent.Context.Compile()

			for event := range agent.accumulator.Generate(
				agent.provider.Generate(params),
			) {
				out <- event
			}

			response := agent.accumulator.String()
			agent.accumulator.Clear()

			if agent.state != AgentStateTerminal {
				agent.Context.AddIteration(response)
				interpreter := NewInterpreter(agent.Context)
				_, agent.state = interpreter.Interpret()

				if agent.state == AgentStateTerminal {
					if agent.container == nil {
						agent.container = tools.NewContainer()
						err := agent.container.Connect(ctx)
						if err != nil {
							errnie.Error(err)
						}
					}
				}

				toolResponse := interpreter.Execute()
				agent.Context.AddIteration(toolResponse)

				if toolResponse != "" {
					out <- &provider.Event{
						Type: provider.EventChunk,
						Text: toolResponse,
					}
				}

				continue
			}

			if agent.state == AgentStateTerminal {
				// Add the command to the context first
				agent.Context.AddIteration(fmt.Sprintf("$ %s", response))

				if strings.Contains(response, "```bash") {
					err := "ERROR: you should not include any Markdown formatting in your response, only respond with valid bash commands, nothing else!"
					agent.Context.AddIteration(err)
					out <- &provider.Event{
						Type: provider.EventChunk,
						Text: err,
					}
					continue
				}

				// Check if we should break out of terminal mode
				if strings.Contains(response, "exit") || strings.Contains(response, `"tool": "break"`) {
					msg := "terminal disconnected"
					agent.Context.AddIteration(msg)
					out <- &provider.Event{
						Type: provider.EventChunk,
						Text: msg,
					}
					agent.state = AgentStateIterating
					continue
				}

				terminalResponse, err := agent.container.RunCommandInteractive(ctx, response)
				if err != nil {
					terminalResponse = fmt.Sprintf("error executing command: %s", err.Error())
				}

				if terminalResponse != "" {
					// The output is already cleaned by RunCommandInteractive
					agent.Context.AddIteration(terminalResponse + "$ ")

					out <- &provider.Event{
						Type: provider.EventChunk,
						Text: terminalResponse,
					}
				}
			}
		}

		agent.Analyze(out)
	}()

	return out
}

func (agent *Agent) Analyze(out chan<- *provider.Event) {
	errnie.Info("analyzing agent", "role", agent.Role)
	// Store the context temporarily.
	mainContext := agent.Context

	// Create a new context for the analyzer.
	v := viper.GetViper()
	analyzerContext := drknow.QuickContext(v.GetString("prompts.templates.systems.analyzer"))
	agent.Context = analyzerContext

	accumulator := stream.NewAccumulator()

	for event := range accumulator.Generate(
		agent.provider.Generate(agent.Context.Compile()),
	) {
		out <- event
	}

	analysis := accumulator.String()
	accumulator.Clear()

	agent.Identity.AddAnalysis(analysis)
	agent.Context = mainContext
}

func (agent *Agent) Validate() bool {
	errnie.Debug("validating agent", "role", agent.Role)

	if agent.Context == nil {
		log.Error("context is nil")
		return false
	}

	if agent.Context.Identity == nil {
		log.Error("identity is nil")
		return false
	}

	if err := agent.Context.Identity.Validate(); err != nil {
		log.Error("identity is invalid", "error", err)
		return false
	}

	if agent.provider == nil {
		log.Error("provider is nil")
		return false
	}

	return true
}
