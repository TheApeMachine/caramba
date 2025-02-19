package tools

import (
	"encoding/json"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

// CoordinatorTool handles team formation and coordination
type CoordinatorTool struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  provider.Parameter `json:"parameters"`
}

// TeamFormation defines the structure for team formation requests
type TeamFormation struct {
	TaskID          string   `json:"task_id"`
	TaskDescription string   `json:"task_description"`
	RequiredRoles   []string `json:"required_roles"`
	TeamSize        int      `json:"team_size"`
	Constraints     []string `json:"constraints,omitempty"`
}

// TeamMember represents a single member in a team
type TeamMember struct {
	ID           string   `json:"id"`
	Role         string   `json:"role"`
	Capabilities []string `json:"capabilities"`
	Status       string   `json:"status"`
}

func NewCoordinatorTool() *CoordinatorTool {
	return &CoordinatorTool{
		Name:        "coordinator",
		Description: "Handles team formation and coordination based on task requirements",
		Parameters: provider.Parameter{
			Properties: map[string]interface{}{
				"task_id": map[string]interface{}{
					"type":        "string",
					"description": "Unique identifier for the task",
				},
				"task_description": map[string]interface{}{
					"type":        "string",
					"description": "Detailed description of the task",
				},
				"required_roles": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "List of required roles for the team",
				},
				"team_size": map[string]interface{}{
					"type":        "integer",
					"description": "Desired team size",
				},
				"constraints": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "Optional constraints for team formation",
				},
			},
			Required: []string{"task_id", "task_description", "required_roles", "team_size"},
		},
	}
}

func (tool *CoordinatorTool) Convert() provider.Tool {
	return provider.Tool{
		Name:        tool.Name,
		Description: tool.Description,
		Parameters:  tool.Parameters,
	}
}

func (tool *CoordinatorTool) Use(agent *ai.Agent, artifact *datura.Artifact) {
	decrypted, err := utils.DecryptPayload(artifact)
	if err != nil {
		panic(err)
	}

	var formation TeamFormation
	err = json.Unmarshal(decrypted, &formation)
	if err != nil {
		panic(err)
	}

	// Create team lead first
	teamLeadIdentity := map[string]interface{}{
		"id":           formation.TaskID + "-lead",
		"name":         "Team Lead for " + formation.TaskID,
		"description":  "Coordinates team for: " + formation.TaskDescription,
		"role":         "teamlead",
		"personality":  "Organized, decisive, and collaborative",
		"motivation":   "To effectively coordinate team members and achieve task goals",
		"beliefs":      "Success comes from clear communication and efficient coordination",
		"goals":        []string{"Coordinate team members", "Ensure task completion", "Maintain team cohesion"},
		"instructions": "Lead team formation and coordination for: " + formation.TaskDescription,
	}

	// Add team lead with coordination tools
	agent.AddAgent(
		"teamlead",
		teamLeadIdentity,
		[]provider.Tool{
			NewTeamTool().Convert(),
			NewMessageTool().Convert(),
			NewCommandTool().Convert(),
		},
	)

	// Create team members for each required role
	for _, role := range formation.RequiredRoles {
		memberIdentity := map[string]interface{}{
			"id":           formation.TaskID + "-" + role,
			"name":         role + " Specialist",
			"description":  role + " specialist for: " + formation.TaskDescription,
			"role":         role,
			"personality":  "Focused, collaborative, and detail-oriented",
			"motivation":   "To excel in " + role + " responsibilities and contribute to team success",
			"beliefs":      "Quality work comes from expertise and collaboration",
			"goals":        []string{"Execute " + role + " tasks", "Collaborate with team", "Maintain high standards"},
			"instructions": "Handle " + role + " responsibilities for: " + formation.TaskDescription,
		}

		agent.AddAgent(
			role,
			memberIdentity,
			[]provider.Tool{
				NewMessageTool().Convert(),
				NewCommandTool().Convert(),
			},
		)
	}
}
