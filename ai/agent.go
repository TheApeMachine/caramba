package ai

import (
	"context"

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
	Context       *Context  `json:"context" jsonschema:"title=Context,description=The context of the agent,required"`
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
	var instructions = "unstructured"

	if agent.params != nil && agent.params.Process != nil {
		instructions = "structured"
	}

	agent.System = NewSystem(agent.Identity, instructions, instructions == "structured")
	agent.params = provider.NewGenerationParams()
	agent.Context = NewContext(agent.System, agent.params)
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

func (agent *Agent) GetRole() string {
	return agent.Identity.Role
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

	agent.Context.Compile()
	iteration := 0

	errnie.Info("%s (%s) generating response", agent.Identity.Name, msg.Role)

	go func() {
		defer close(out)

		for iteration < agent.MaxIterations {
			scratchpad := agent.Context.GetScratchpad()

			for event := range agent.provider.Generate(ctx, agent.Context.Compile()) {
				if event.Type == provider.EventChunk && event.Text != "" {
					scratchpad.Append(event)
				}

				if event.Type == provider.EventToolCall {
					scratchpad.ToolCall(event)
				}

				out <- event
			}

			if agent.Context.Done() {
				break
			}

			iteration++
		}
	}()

	return out
}
