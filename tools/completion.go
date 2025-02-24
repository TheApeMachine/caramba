package tools

import (
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
)

/*
CompletionTool which is used by agents to indicate task completion,
and break out of their iteration loop.
*/
type CompletionTool struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  provider.Parameter `json:"parameters"`
}

/*
NewCompletionTool creates a new CompletionTool.
*/
func NewCompletionTool() *CompletionTool {
	return &CompletionTool{
		Name:        "indicate_completion",
		Description: "Use this tool to indicate that the current task or iteration is complete",
		Parameters: provider.Parameter{
			Type: "object",
			Properties: map[string]interface{}{
				"is_complete": map[string]interface{}{
					"type":        "boolean",
					"description": "Whether the task is complete",
				},
				"reason": map[string]interface{}{
					"type":        "string",
					"description": "Reason for completion",
				},
			},
			Required: []string{"is_complete"},
		},
	}
}

/*
Convert converts the CompletionTool to a provider.Tool.
*/
func (tool *CompletionTool) Convert() provider.Tool {
	return provider.Tool{
		Name:        tool.Name,
		Description: tool.Description,
		Parameters:  tool.Parameters,
	}
}

/*
Use handles the execution of the completion tool.
In this case we only have to set the state of the agent to
break the iteration loop.
*/
func (tool *CompletionTool) Use(agent *ai.Agent, artifact *datura.Artifact) {
	agent.State = ai.AgentStateCompleted
}
