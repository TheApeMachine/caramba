package tools

import (
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/system"
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
	From       string      `json:"from"`
	To         string      `json:"to"`
	Topic      string      `json:"topic"`
	Content    string      `json:"content"`
	Context    string      `json:"context,omitempty"`
	References []string    `json:"references,omitempty"`
	Priority   int         `json:"priority,omitempty"`
	Expiration string      `json:"expiration,omitempty"`
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
	q := system.NewQueue()
	q.SendMessage(artifact)
	agent.AddContext("<message>SENT</message>")
}
