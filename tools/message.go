package tools

import (
	"encoding/json"
	"strings"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/system"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

// MessageType defines different types of team messages
type MessageType string

const (
	MessageTypeTask       MessageType = "task"
	MessageTypeDiscussion MessageType = "discussion"
	MessageTypeDecision   MessageType = "decision"
	MessageTypeStatus     MessageType = "status"
	MessageTypeQuestion   MessageType = "question"
	MessageTypeResponse   MessageType = "response"
)

// MessageTool handles structured communication between agents
type MessageTool struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  provider.Parameter `json:"parameters"`
}

// MessageParams defines the structure for message parameters
type MessageParams struct {
	From       string   `json:"from"`
	To         string   `json:"to"`
	Topic      string   `json:"topic"`
	Content    string   `json:"content"`
	Context    string   `json:"context,omitempty"`
	References []string `json:"references,omitempty"`
	Priority   int      `json:"priority,omitempty"`
	Expiration string   `json:"expiration,omitempty"`
}

func NewMessageTool() *MessageTool {
	return &MessageTool{
		Name:        "message",
		Description: "Enables structured communication between team members",
		Parameters: provider.Parameter{
			Properties: map[string]interface{}{
				"from": map[string]interface{}{
					"type":        "string",
					"description": "ID of the sending agent",
				},
				"to": map[string]interface{}{
					"type":        "string",
					"description": "ID of the receiving agent, name of the topic, or broadcast",
				},
				"subject": map[string]interface{}{
					"type":        "string",
					"description": "Subject of the message",
				},
				"content": map[string]interface{}{
					"type":        "string",
					"description": "Main content of the message",
				},
				"context": map[string]interface{}{
					"type":        "string",
					"description": "Additional context for the message",
				},
				"references": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "References to other messages or resources",
				},
				"priority": map[string]interface{}{
					"type":        "integer",
					"description": "Priority level of the message (1-5)",
					"minimum":     1,
					"maximum":     5,
				},
				"expiration": map[string]interface{}{
					"type":        "string",
					"description": "When this message should be considered expired",
				},
			},
			Required: []string{"from", "to", "topic", "content"},
		},
	}
}

func (tool *MessageTool) Convert() provider.Tool {
	return provider.Tool{
		Name:        tool.Name,
		Description: tool.Description,
		Parameters:  tool.Parameters,
	}
}

func (tool *MessageTool) Use(agent *ai.Agent, artifact *datura.Artifact) {
	errnie.Info("🔨 *MessageTool.Use")

	q := system.NewQueue()

	// Decrypt and parse the message parameters
	decrypted, err := utils.DecryptPayload(artifact)
	if err != nil {
		agent.AddContext("<error>Failed to decrypt message payload</error>")
		return
	}

	var params MessageParams
	if err := json.Unmarshal(decrypted, &params); err != nil {
		agent.AddContext("<error>Failed to parse message parameters</error>")
		return
	}

	out := strings.Join(
		[]string{
			"<error>Target agent not found: " + params.To + "</error>",
			"<available_agents>",
			"\t<agent>",
			"\t\t<id>" + agent.Identity.ID + "</id>",
			"\t\t<name>" + agent.Identity.Name + "</name>",
			"\t\t<role>" + agent.Identity.Role + "</role>",
			"\t</agent>",
			"</available_agents>",
		},
		"\n",
	)

	// Check if target agent exists
	if params.To != "broadcast" && params.To != "" {
		targetAgent := q.GetAgent(params.To)
		if targetAgent == nil {
			agent.AddContext(out)
			return
		}
	}

	q.SendMessage(artifact)
	agent.AddContext("<message>SENT</message>")
}
