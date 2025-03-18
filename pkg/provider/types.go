package provider

import (
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type MessageRole string

const (
	MessageRoleSystem    MessageRole = "system"
	MessageRoleUser      MessageRole = "user"
	MessageRoleDeveloper MessageRole = "developer"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleTool      MessageRole = "tool"
)

type Message struct {
	Role    MessageRole `json:"role"`
	Name    string      `json:"name"`
	Content string      `json:"content"`
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Function struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  Parameters `json:"parameters"`
}

type Parameters struct {
	Type       string     `json:"type"`
	Properties []Property `json:"properties"`
	Required   []string   `json:"required"`
}

type Property struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
	Enum        []any  `json:"enum"`
}

type ResponseFormat struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Schema      any    `json:"schema"`
	Strict      bool   `json:"strict"`
}

type Params struct {
	Model            string         `json:"model"`
	Messages         []Message      `json:"messages"`
	Tools            []Tool         `json:"tools"`
	ResponseFormat   ResponseFormat `json:"response_format"`
	Temperature      float64        `json:"temperature"`
	TopP             float64        `json:"top_p"`
	TopK             float64        `json:"top_k"`
	FrequencyPenalty float64        `json:"frequency_penalty"`
	PresencePenalty  float64        `json:"presence_penalty"`
	MaxTokens        int            `json:"max_tokens"`
	Stream           bool           `json:"stream"`
}

func (params *Params) Marshal() []byte {
	json, err := json.Marshal(params)

	if errnie.Error(err) != nil {
		return nil
	}

	return json
}

func (params *Params) Unmarshal(data []byte) {
	errnie.Debug("provider.Params.Unmarshal")
	errnie.Error(json.Unmarshal(data, params))
}
