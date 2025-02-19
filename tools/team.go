package tools

import (
	"encoding/json"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

type TeamTool struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  provider.Parameter `json:"parameters"`
}

func NewTeamTool() *TeamTool {
	return &TeamTool{
		Name:        "team",
		Description: "This tool is used to create a new team.",
		Parameters: provider.Parameter{
			Properties: map[string]interface{}{
				"id": map[string]interface{}{
					"type":        "string",
					"description": "The unique identifier for the team",
				},
				"name": map[string]interface{}{
					"type":        "string",
					"description": "The name of the team",
				},
				"description": map[string]interface{}{
					"type":        "string",
					"description": "A description of the team",
				},
				"goals": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "The goals of the team",
				},
				"instructions": map[string]interface{}{
					"type":        "string",
					"description": "The instructions of the team",
				},
				"discussion_topic": map[string]interface{}{
					"type":        "string",
					"description": "The topic to discuss",
				},
				"participants": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "The participants in the discussion",
				},
			},
			Required: []string{"id", "name", "description", "goals", "instructions"},
		},
	}
}

func (tool *TeamTool) Convert() provider.Tool {
	return provider.Tool{
		Name:        tool.Name,
		Description: tool.Description,
		Parameters:  tool.Parameters,
	}
}

type TeamParams struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	Goals        []string `json:"goals"`
	Instructions string   `json:"instructions"`
}

func (tool *TeamTool) Use(agent *ai.Agent, artifact *datura.Artifact) {
	decrypted, err := utils.DecryptPayload(artifact)

	if err != nil {
		panic(err)
	}

	var params TeamParams
	err = json.Unmarshal(decrypted, &params)

	if err != nil {
		panic(err)
	}

	identity := map[string]interface{}{
		"id":           params.ID + "-teamlead",
		"name":         params.Name + " Team Lead",
		"description":  "Team lead for: " + params.Description,
		"role":         "teamlead",
		"personality":  "Organized, inclusive, and results-driven",
		"motivation":   "To create high-performing teams that leverage each team's unique strengths",
		"beliefs":      "Success comes from well-structured teams working in harmony with clear goals",
		"goals":        params.Goals,
		"instructions": params.Instructions,
	}

	agent.AddAgent(
		params.ID+"-teamlead",
		identity,
		[]provider.Tool{
			NewAgentTool().Convert(),
			NewCommandTool().Convert(),
		},
	)
}
