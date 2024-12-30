package ai

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/process/reasoning"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
Agent wraps the requirements and functionality to turn a prompt and
response sequence into behavior. It enhances the default functionality
of a large language model, by adding optional structured responses and
tool calling.
*/
type Agent struct {
	System        *System   `json:"system" jsonschema:"title=System,description=The system prompt for the agent,required"`
	Identity      *Identity `json:"identity" jsonschema:"title=Identity,description=The identity of the agent,required"`
	MaxIterations int       `json:"max_iterations" jsonschema:"title=Max Iterations,description=The maximum number of iterations to perform,required"`
	params        *provider.GenerationParams
	provider      provider.Provider
}

/*
NewAgent creates a new Agent instance.
*/
func NewAgent(ctx context.Context, role string, maxIterations int) *Agent {
	return &Agent{
		Identity:      NewIdentity(ctx, role),
		provider:      provider.NewBalancedProvider(),
		MaxIterations: maxIterations,
	}
}

/*
GenerateSchema adheres the Agent type to the Tool interface, meaning Agent
is also a Tool. This allows agents to create new agents for task delegation,
provided that have been given access to the Agent tool.
*/
func (agent *Agent) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Agent]()
}

func (agent *Agent) Initialize() {
	agent.System = NewSystem(agent.Identity, (agent.params != nil && agent.params.Process != nil))
	agent.params = provider.NewGenerationParams()
	agent.params.Thread.AddMessage(
		provider.NewMessage(provider.RoleSystem, agent.System.String()),
	)
}

func (agent *Agent) AddTools(tools ...provider.Tool) {
	if agent.params == nil {
		agent.Initialize()
	}

	agent.params.Tools = append(agent.params.Tools, tools...)
}

/*
AddProcess activates structured outputs for the agent, in which it
will follow the specific jsonschema defined by the process.
*/
func (agent *Agent) AddProcess(process provider.Process) {
	agent.params.Process = process

	// We have to reinitialize the agent, because the system prompt
	// is different when structured outputs are enabled.
	agent.Initialize()
}

/*
RemoveProcess deactivates structured outputs for the agent, and it
will revert back to generating freeform text.
*/
func (agent *Agent) RemoveProcess() {
	agent.params.Process = nil
}

/*
Generate calls the underlying provider to have a Large Language Model
generate text for the agent.
*/
func (agent *Agent) Generate(ctx context.Context, msg *provider.Message) <-chan provider.Event {
	out := make(chan provider.Event)

	if agent.params == nil {
		agent.Initialize()
	}

	// Only add non-empty messages to the thread
	if msg.Content == "" {
		errnie.Warn("empty message received %s (%s)", agent.Identity.Name, msg.Role)
		return nil
	}

	errnie.Info("%s (%s) generating response", agent.Identity.Name, msg.Role)

	agent.params.Thread.AddMessage(msg)
	iteration := 0

	go func() {
		defer close(out)

		for iteration < agent.MaxIterations {
			// Only add status if we have a valid status template
			if status := agent.getStatus(iteration); status != "" {
				agent.params.Thread.AddMessage(provider.NewMessage(provider.RoleUser, status))
			}

			// Accumulate the response chunks into a single message
			accumulator := agent.makeAccumulator(iteration)

			for event := range agent.provider.Generate(ctx, agent.params) {
				if event.Type == provider.EventChunk && event.Text != "" {
					accumulator.Append(event.Text)
				}

				out <- event
			}

			// Only add non-empty messages to the thread
			if accumulator.Content != "" {
				agent.params.Thread.AddMessage(agent.closeAccumulator(accumulator))
			}

			proc := &reasoning.Process{}

			if agent.params.Process != nil && errnie.Error(
				json.Unmarshal([]byte(accumulator.Content), agent.params.Process),
			) != nil {
				if proc.FinalAnswer != "" {
					break
				}
			}

			iteration++
		}
	}()

	return out
}

/*
makeAccumulator creates a new Message with the appropriate tags for
the agent's response.
*/
func (agent *Agent) makeAccumulator(iteration int) *provider.Message {
	return provider.NewMessage(
		provider.RoleAssistant,
		"\t<response agent="+agent.Identity.Name+" role="+agent.Identity.Role+" iteration="+strconv.Itoa(iteration)+" >",
	)
}

/*
closeAccumulator closes the accumulator message, and adds the appropriate
tags to the end of the message.
*/
func (agent *Agent) closeAccumulator(accumulator *provider.Message) *provider.Message {
	accumulator.Append("\t</response>")
	return accumulator
}

// Split status creation from adding to thread
func (agent *Agent) getStatus(iteration int) string {
	v := viper.GetViper()
	template := v.GetString("prompts.template.status")
	if template == "" {
		return ""
	}

	return utils.ReplaceWith(
		template,
		[][]string{
			{"iteration", strconv.Itoa(iteration + 1)},
			{"remaining", strconv.Itoa(agent.MaxIterations - iteration)},
		},
	)
}
