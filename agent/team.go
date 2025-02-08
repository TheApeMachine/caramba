package agent

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

type Team struct {
	Tool         string `json:"tool" jsonschema:"title=Tool,description=The tool to use for the team,enum=team,required"`
	SystemPrompt string `json:"system_prompt" jsonschema:"title=System Prompt,description=The system prompt to use for the team lead,required"`
	agents       []*Config
}

func NewTeam(system, name, role string) *Team {
	return &Team{
		Tool:         "team",
		SystemPrompt: system,
		agents:       []*Config{},
	}
}

func (team *Team) GenerateSchema() interface{} {
	return utils.GenerateSchema[Team]()
}

func (team *Team) Name() string {
	return team.Tool
}

func (team *Team) Description() string {
	return team.SystemPrompt
}

func (team *Team) Use(ctx context.Context, input map[string]interface{}) (string, error) {
	return utils.JoinWith("\n",
		"",
		"  [TEAMLEAD]",
		"    Name  : "+team.Name(),
		"    Status: READY FOR DUTY",
		"  [/TEAMLEAD]",
	), nil
}

func (team *Team) Connect(ctx context.Context, rwc io.ReadWriteCloser) error {
	return nil
}
