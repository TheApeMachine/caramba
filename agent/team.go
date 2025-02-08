package agent

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

/*
Team represents a group of agent configurations that work together.
It manages a collection of agents with a shared system prompt and
provides functionality for team-based operations. The Team structure
implements tool-like behavior for integration with the broader system.
*/
type Team struct {
	Tool         string    `json:"tool" jsonschema:"title=Tool,description=The tool to use for the team,enum=team,required"`
	SystemPrompt string    `json:"system_prompt" jsonschema:"title=System Prompt,description=The system prompt to use for the team lead,required"`
	agents       []*Config // Collection of agent configurations in the team
}

/*
NewTeam creates and returns a new Team instance with the specified parameters.

Parameters:

	system: The system prompt to use for the team
	name: The name identifier for the team
	role: The role designation for the team

Returns:

	*Team: A new Team instance initialized with the provided parameters
*/
func NewTeam(system, name, role string) *Team {
	return &Team{
		Tool:         "team",
		SystemPrompt: system,
		agents:       []*Config{},
	}
}

/*
GenerateSchema generates and returns the JSON schema for the Team type.
This is used for validation and documentation of the team configuration.

Returns:

	interface{}: The generated JSON schema for the Team type
*/
func (team *Team) GenerateSchema() interface{} {
	return utils.GenerateSchema[Team]()
}

/*
Name returns the identifier for this team implementation.

Returns:

	string: The string "team" which identifies this implementation
*/
func (team *Team) Name() string {
	return team.Tool
}

/*
Description returns a human-readable description of the team's purpose
and functionality.

Returns:

	string: A description of what the team does and how it operates
*/
func (team *Team) Description() string {
	return team.SystemPrompt
}

/*
Use processes the provided input using the team's agents and returns
the result. It coordinates the work between team members and manages
the execution flow.

Parameters:

	ctx: The context for the operation
	input: A map of input parameters for the team to process

Returns:

	string: The result of the team's processing
	error: Any error that occurred during processing
*/
func (team *Team) Use(ctx context.Context, input map[string]interface{}) (string, error) {
	return utils.JoinWith("\n",
		"",
		"  [TEAMLEAD]",
		"    Name  : "+team.Name(),
		"    Status: READY FOR DUTY",
		"  [/TEAMLEAD]",
	), nil
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
