package tools

import (
	"encoding/json"

	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

type CommandTool struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  provider.Parameter `json:"parameters"`
}

func NewCommandTool() *CommandTool {
	return &CommandTool{
		Name:        "command",
		Description: "This tool is used to execute commands.",
		Parameters: provider.Parameter{
			Properties: map[string]interface{}{
				"command": map[string]interface{}{
					"type":        "string",
					"description": "The command to execute",
				},
				"args": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "The arguments for the command",
				},
			},
			Required: []string{"command"},
		},
	}
}

func (tool *CommandTool) Convert() provider.Tool {
	return provider.Tool{
		Name:        tool.Name,
		Description: tool.Description,
		Parameters:  tool.Parameters,
	}
}

func (tool *CommandTool) Use(agent interface{}, artifact *datura.Artifact) {
	decrypted, err := utils.DecryptPayload(artifact)
	if err != nil {
		panic(err)
	}

	var params struct {
		Command string   `json:"command"`
		Args    []string `json:"args"`
	}

	if err := json.Unmarshal(decrypted, &params); err != nil {
		panic(err)
	}

	// TODO: Implement command execution logic
	// This should be implemented based on your specific needs
	// Consider security implications and implement appropriate safeguards
}
