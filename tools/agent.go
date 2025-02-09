package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/agent"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/utils"
)

/*
Agent represents an agent with specific roles and capabilities.
*/
type Agent struct {
	Tool string `json:"tool" jsonschema:"title=Tool,description=The tool to use for the agent,enum=agent,required"`
	Args struct {
		AgentName    string `json:"agent_name" jsonschema:"title=Agent Name,description=The name of the agent,required"`
		Role         string `json:"role" jsonschema:"title=Role,description=The role of this agent,required"`
		SystemPrompt string `json:"system_prompt" jsonschema:"title=System Prompt,description=The system prompt to use for the agent,required"`
	} `json:"args" jsonschema:"title=Arguments,description=The arguments to pass to the agent,required"`
}

/*
NewAgent creates and returns a new Agent instance with the specified parameters.

Returns:

	*Agent: A new Agent instance
*/
func NewAgent() *Agent {
	return &Agent{}
}

/*
Use creates a new agent, which belongs to the agent being passed in.

Parameters:

	input: A map of input parameters for the agent to process
	generators: A list of agents.

Returns:

	string: The result of the agent's processing
*/
func (agentTool *Agent) Use(
	accumulator *stream.Accumulator,
	input map[string]any,
	generators ...*agent.Generator,
) *stream.Accumulator {
	out := make(chan *provider.Event)

	go func() {
		defer close(out)

		newAgent := agent.NewGenerator(
			agent.NewConfig(
				input["agent_name"].(string),
				input["role"].(string),
				input["system_prompt"].(string),
				NewToolset().String(),
			),
			provider.NewBalancedProvider(),
		)

		for _, generator := range generators {
			generator.Agents[input["agent_name"].(string)] = newAgent
		}

		accumulator.Append(
			utils.QuickWrap("AGENT", utils.JoinWith("\n",
				"NAME  : "+input["agent_name"].(string),
				"ROLE  : "+input["role"].(string),
				"STATUS: READY",
			), 1),
		)
	}()

	return accumulator
}

/*
Name returns the identifier for this agent implementation.

Returns:

	string: The agent's name
*/
func (agentTool *Agent) Name() string {
	return agentTool.Args.AgentName
}

/*
Description returns a human-readable description of the agent's purpose
and functionality.

Returns:

	string: A description of what the agent does and how it operates
*/
func (agentTool *Agent) Description() string {
	return agentTool.Args.SystemPrompt
}

/*
Connect establishes a connection for the agent using the provided
ReadWriteCloser. This allows the agent to communicate with external
systems or interfaces.

Parameters:

	ctx: The context for the connection operation
	rwc: The ReadWriteCloser to use for communication

Returns:

	error: Any error that occurred during connection setup
*/
func (agentTool *Agent) Connect(ctx context.Context, rwc io.ReadWriteCloser) error {
	return nil
}

func (agentTool *Agent) GenerateSchema() any {
	return utils.GenerateSchema[Agent]()
}
