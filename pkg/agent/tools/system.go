package tools

import "context"

type System struct {
	Command string `json:"command"`
}

func NewSystem() *System {
	return &System{}
}

func (system *System) Name() string {
	return "system"
}

func (system *System) Description() string {
	return "Execute a system command"
}

func (system *System) Execute(ctx context.Context, args map[string]any) (any, error) {
	return system.Command, nil
}

func (system *System) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command": map[string]any{
				"type":        "string",
				"description": "The command to execute",
				"enum":        []string{"break"},
			},
		},
		"required": []string{"command"},
	}
}
