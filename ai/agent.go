package ai

import (
	"context"
	"fmt"
	"io"
	"regexp"
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

		shouldBreak := false
		cycle := 0

		// Channel to signal command completion
		cmdDone := make(chan string)

		// Regex to match the bash prompt pattern
		promptRegex := regexp.MustCompile(`user@[^:]+:[^$]+\$`)

		// Add the user prompt, and wrap it in some tags, to keep clarity along an ever
		// growing complexity in the current context.
		msg.Content = utils.QuickWrap(
			"USER PROMPT",
			msg.Content,
			1,
		)

		agent.Context.AddMessage(msg)

		for !shouldBreak {
			agent.state = AgentStateGenerating

			cycle++
			if cycle >= agent.MaxIterations {
				shouldBreak = true
			}

			// Compile the context, which will add the current iteration message
			// to the thread.
			compiled := agent.Context.Compile(cycle, agent.MaxIterations)

			// Let's make sure to reset the accumulator.
			agent.accumulator.Clear()

			for event := range agent.accumulator.Generate(
				ctx,
				agent.provider.Generate(
					ctx,
					compiled,
				),
			) {
				out <- event
			}

			// Get only this iteration's response
			response := agent.accumulator.String()

			// Add the response to context
			agent.Context.AddMessage(
				provider.NewMessage(
					provider.RoleAssistant,
					response,
				),
			)

			// Process commands in the response using the interpreter
			if agent.bridge == nil {
				interpreter := NewInterpreter(agent.Context)
				interpreter, agent.state = interpreter.Interpret()
				agent.bridge = interpreter.Execute()

				if agent.bridge != nil && agent.state == AgentStateTerminal {
					var outputBuffer strings.Builder
					buf := make([]byte, 4096)
					readDone := make(chan bool)

					// Start the continuous reader
					go func() {
						for {
							n, err := agent.bridge.Read(buf)
							if err != nil {
								if err != io.EOF {
									fmt.Printf("Error reading from container bridge: %v\n", err)
								}
								break
							}
							if n > 0 {
								output := string(buf[:n])
								outputBuffer.WriteString(output)
								fmt.Print(output)

								if promptRegex.MatchString(output) {
									result := outputBuffer.String()
									if strings.TrimSpace(result) != "" {
										cmdDone <- strings.TrimSpace(result)
									}
									outputBuffer.Reset()
								}
							}
						}
						close(readDone)
					}()

					// Wait for initial prompt
					select {
					case output := <-cmdDone:
						agent.Context.AddMessage(provider.NewMessage(
							provider.RoleAssistant,
							output,
						))
						// Return immediately after getting initial prompt to let agent respond
						return
					case <-ctx.Done():
						log.Warn("Context cancelled while waiting for initial prompt")
						return
					}
				}
			} else if agent.state == AgentStateTerminal {
				agent.bridge.Write([]byte(response + "\n"))

				select {
				case output := <-cmdDone:
					agent.Context.AddMessage(provider.NewMessage(
						provider.RoleAssistant,
						output,
					))
				case <-ctx.Done():
					log.Warn("Context cancelled while waiting for command output")
				}
			}

			if strings.Contains(
				strings.ToLower(response),
				"<break",
			) {
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
