package provider

import (
	"fmt"

	"github.com/theapemachine/amsh/utils"
)

// Event represents different types of provider events
type Event struct {
	TeamID  string
	AgentID string
	Type    EventType
	Content string
	Error   error
}

type EventType int

const (
	EventToken EventType = iota
	EventToolCall
	EventFunctionCall
	EventError
	EventDone
)

type Message struct {
	Role     string                 `json:"role"`
	Content  string                 `json:"content"`
	Name     string                 `json:"name,omitempty"`
	Function map[string]interface{} `json:"function,omitempty"`
}

func mapMessages(messages []Message) []string {
	result := make([]string, len(messages))
	for i, message := range messages {
		if message.Function != nil {
			result[i] = fmt.Sprintf("%s: %s", message.Name, message.Function)
		} else {
			result[i] = fmt.Sprintf("%s: %s", message.Role, message.Content)
		}
	}
	return result
}

type GenerationParams struct {
	Messages               []Message
	Temperature            float64
	PresencePenalty        float64
	FrequencyPenalty       float64
	TopP                   float64
	TopK                   int
	Interestingness        float64
	InterestingnessHistory []float64
}

func (params GenerationParams) String() string {
	return utils.JoinWith("\n\n",
		utils.JoinWith("\n", mapMessages(params.Messages)...),
		utils.JoinWith(
			"\n",
			fmt.Sprintf("Temperature: %f", params.Temperature),
			fmt.Sprintf("PresencePenalty: %f", params.PresencePenalty),
			fmt.Sprintf("FrequencyPenalty: %f", params.FrequencyPenalty),
			fmt.Sprintf("TopP: %f", params.TopP),
			fmt.Sprintf("TopK: %d", params.TopK),
		),
	)
}

// Provider defines the interface for AI providers
type Provider interface {
	Generate(GenerationParams) <-chan Event
}
