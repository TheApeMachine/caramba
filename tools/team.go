package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/agent"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
TeamMember represents a member of a team, which can be either an agent or another team.
This allows for hierarchical team structures.
*/
type TeamMember interface {
	Name() string
	Description() string
	Use(input map[string]any) string
	Connect(ctx context.Context, rwc io.ReadWriteCloser) error
}

/*
Team represents a group of agent configurations that work together.
It manages a collection of agents with a shared system prompt and
provides functionality for team-based operations. The Team structure
implements tool-like behavior for integration with the broader system.

The coordinator creates the team with team configuration, and the team lead
is assigned programmatically based on the teamLeadRole.
*/
type Team struct {
	// Configuration (what coordinator sets)
	Tool string `json:"tool" jsonschema:"title=Tool,description=The tool to use for the team,enum=team,required"`
	Args struct {
		TeamName     string `json:"team_name" jsonschema:"title=Team Name,description=The name of the team,required"`
		SystemPrompt string `json:"system_prompt" jsonschema:"title=System Prompt,description=The system prompt to use for the team,required"`
	} `json:"args" jsonschema:"title=Arguments,description=The arguments to pass to the tool,required"`
}

/*
NewTeam creates and returns a new Team instance with the specified parameters.

Returns:

	*Team: A new Team instance
*/
func NewTeam() *Team {
	return &Team{}
}

/*
GenerateSchema generates and returns the JSON schema for the Team type.
This is used for validation and documentation of the team configuration.

Returns:

	interface{}: The generated JSON schema for the Team type
*/
func (team *Team) GenerateSchema() any {
	return utils.GenerateSchema[Team]()
}

/*
Name returns the identifier for this team implementation.

Returns:

	string: The string "team" which identifies this implementation
*/
func (team *Team) Name() string {
	return team.Args.TeamName
}

/*
Description returns a human-readable description of the team's purpose
and functionality.

Returns:

	string: A description of what the team does and how it operates
*/
func (team *Team) Description() string {
	return team.Args.SystemPrompt
}

/*
Use handles the message flow: coordinator → team → team lead → agents

Parameters:

	input: A map of input parameters for the team to process

Returns:

	string: The result of the team's processing
*/
func (team *Team) Use(
	accumulator *stream.Accumulator, input map[string]any, generators ...*agent.Generator,
) *stream.Accumulator {
	errnie.Debug("using team", "team", input["team_name"])

	newAgent := agent.NewGenerator(
		agent.NewConfig(
			input["system_prompt"].(string),
			"teamlead",
			input["team_name"].(string),
			NewToolset(&Agent{}).String(),
		),
		provider.NewBalancedProvider(),
	)

	for _, generator := range generators {
		generator.Agents[input["team_name"].(string)] = newAgent
	}

	accumulator.Append(
		utils.QuickWrap("TEAM", utils.JoinWith("\n",
			"NAME  : "+input["team_name"].(string),
			"STATUS: READY",
		), 1),
	)

	return accumulator
}

/*
Connect establishes a connection for the team using the provided
ReadWriteCloser. This allows the team to communicate with external
systems or interfaces.

Parameters:

	ctx: The context for the connection operation
	rwc: The ReadWriteCloser to use for communication

Returns:

	error: Any error that occurred during connection setup
*/
func (team *Team) Connect(ctx context.Context, rwc io.ReadWriteCloser) error {
	return nil
}
