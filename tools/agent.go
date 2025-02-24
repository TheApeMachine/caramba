package tools

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/system"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
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

func (tool *AgentTool) Use(agent *ai.Agent, artifact *datura.Artifact) {
	errnie.Info("🔨 *AgentTool.Use")

	decrypted, err := utils.DecryptPayload(artifact)

	if err != nil {
		panic(err)
	}

	var wrapper struct {
		Arguments string `json:"arguments"`
		Name      string `json:"name"`
		Index     int    `json:"Index"`
	}

	err = json.Unmarshal(decrypted, &wrapper)

	if err != nil {
		panic(err)
	}

	var params map[string]interface{}
	err = json.Unmarshal([]byte(wrapper.Arguments), &params)

	if err != nil {
		panic(err)
	}

	// Convert goals from []interface{} to []string
	goalsInterface, ok := params["goals"].([]interface{})

	if !ok {
		panic("goals is not an array")
	}

	goals := make([]string, len(goalsInterface))
	for i, v := range goalsInterface {
		goals[i], ok = v.(string)
		if !ok {
			panic("goal item is not a string")
		}
	}

	// Create identity map
	identity := ai.Identity{
		ID:           params["id"].(string),
		Name:         params["name"].(string),
		Description:  params["description"].(string),
		Role:         params["role"].(string),
		Personality:  params["personality"].(string),
		Motivation:   params["motivation"].(string),
		Beliefs:      params["beliefs"].(string),
		Goals:        goals,
		Instructions: params["instructions"].(string),
	}

	role := params["role"].(string)

	if agent.Agents[role] == nil {
		agent.Agents[role] = make([]*ai.Agent, 0)
	}

	delegate := ai.NewAgent(
		&identity,
		[]provider.Tool{
			NewCommandTool().Convert(),
			NewCompletionTool().Convert(),
			NewMessageTool().Convert(),
			NewAgentTool().Convert(),
		},
	)

	agent.Agents[role] = append(agent.Agents[role], delegate)
	system.NewQueue().AddAgent(delegate)

	out := strings.Join(
		[]string{
			"<agent>",
			"\t<id>" + identity.ID + "</id>",
			"\t<name>" + identity.Name + "</name>",
			"\t<role>" + identity.Role + "</role>",
			"\t<status>CREATED</status>",
			"</agent>",
		},
		"\n",
	)

	fmt.Println(out)

	agent.Params.Messages = append(agent.Params.Messages, provider.Message{
		Role:    "assistant",
		Content: out,
	})
}
