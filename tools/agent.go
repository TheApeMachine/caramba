package tools

import (
	"encoding/json"

	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

type AgentTool struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  provider.Parameter `json:"parameters"`
}

func NewAgentTool() *AgentTool {
	return &AgentTool{
		Name:        "agent",
		Description: "This tool is used to create a new agent.",
		Parameters: provider.Parameter{
			Properties: map[string]interface{}{
				"id": map[string]interface{}{
					"type":        "string",
					"description": "The unique identifier for the identity",
				},
				"name": map[string]interface{}{
					"type":        "string",
					"description": "The name of the identity",
				},
				"description": map[string]interface{}{
					"type":        "string",
					"description": "A description of the identity",
				},
				"role": map[string]interface{}{
					"type":        "string",
					"description": "The role of the identity",
				},
				"personality": map[string]interface{}{
					"type":        "string",
					"description": "The personality of the identity",
				},
				"motivation": map[string]interface{}{
					"type":        "string",
					"description": "The motivation of the identity",
				},
				"beliefs": map[string]interface{}{
					"type":        "string",
					"description": "The beliefs of the identity",
				},
				"goals": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "The goals of the identity",
				},
				"instructions": map[string]interface{}{
					"type":        "string",
					"description": "The instructions of the identity",
				},
			},
			Required: []string{"id", "name", "description", "role", "personality", "motivation", "beliefs", "goals", "instructions"},
		},
	}
}

func (tool *AgentTool) Convert() provider.Tool {
	return provider.Tool{
		Name:        tool.Name,
		Description: tool.Description,
		Parameters:  tool.Parameters,
	}
}

type AgentCreator interface {
	AddAgent(role string, identity interface{}, tools []provider.Tool)
}

func (tool *AgentTool) Use(agent interface{}, artifact *datura.Artifact) {
	decrypted, err := utils.DecryptPayload(artifact)
	if err != nil {
		panic(err)
	}

	var params map[string]interface{}
	err = json.Unmarshal(decrypted, &params)
	if err != nil {
		panic(err)
	}

	// Extract goals as []string
	goalsInterface := params["goals"].([]interface{})
	goals := make([]string, len(goalsInterface))
	for i, g := range goalsInterface {
		goals[i] = g.(string)
	}

	// Create identity map
	identity := map[string]interface{}{
		"id":           params["id"].(string),
		"name":         params["name"].(string),
		"description":  params["description"].(string),
		"role":         params["role"].(string),
		"personality":  params["personality"].(string),
		"motivation":   params["motivation"].(string),
		"beliefs":      params["beliefs"].(string),
		"goals":        goals,
		"instructions": params["instructions"].(string),
	}

	if creator, ok := agent.(AgentCreator); ok {
		creator.AddAgent(
			params["role"].(string),
			identity,
			[]provider.Tool{
				NewCommandTool().Convert(),
				NewMessageTool().Convert(),
			},
		)
	}
}
