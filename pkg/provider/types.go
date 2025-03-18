package provider

import (
	"encoding/json"

	"github.com/mark3labs/mcp-go/mcp"
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

type OptionMessage func(*Message)

func NewMessage(opts ...OptionMessage) *Message {
	msg := &Message{}

	for _, opt := range opts {
		opt(msg)
	}

	return msg
}

func WithSystemRole(content string) OptionMessage {
	return func(msg *Message) { msg.Role = MessageRoleSystem; msg.Content = content }
}

func WithDeveloperRole(name string, content string) OptionMessage {
	return func(msg *Message) { msg.Role = MessageRoleDeveloper; msg.Name = name; msg.Content = content }
}

func WithUserRole(name string, content string) OptionMessage {
	return func(msg *Message) { msg.Role = MessageRoleUser; msg.Name = name; msg.Content = content }
}

func WithAssistantRole(name string, content string) OptionMessage {
	return func(msg *Message) { msg.Role = MessageRoleAssistant; msg.Name = name; msg.Content = content }
}

func WithToolRole(name string, content string) OptionMessage {
	return func(msg *Message) { msg.Role = MessageRoleTool; msg.Name = name; msg.Content = content }
}

var RegisteredTools = []any{}

func RegisterTool(name string) {
	RegisteredTools = append(RegisteredTools, name)
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type OptionTool func(*Tool)

func NewTool(opts ...OptionTool) *Tool {
	tool := &Tool{
		Type: "function",
	}

	for _, opt := range opts {
		opt(tool)
	}

	return tool
}

func (tool *Tool) ToMCP() *mcp.Tool {
	options := []mcp.ToolOption{
		mcp.WithDescription(tool.Function.Description),
	}

	for _, property := range tool.Function.Parameters.Properties {
		switch property.Type {
		case "string":
			options = append(options, mcp.WithString(
				property.Name,
				mcp.Description(property.Description),
			))
		case "number":
			options = append(options, mcp.WithNumber(
				property.Name,
				mcp.Description(property.Description),
			))
		case "boolean":
			options = append(options, mcp.WithBoolean(
				property.Name,
				mcp.Description(property.Description),
			))
		}
	}

	mt := mcp.NewTool(
		tool.Function.Name,
		options...,
	)

	return &mt
}

func WithFunction(name string, description string) OptionTool {
	return func(tool *Tool) {
		tool.Function = Function{
			Name:        name,
			Description: description,
			Parameters: Parameters{
				Type:       "object",
				Properties: []Property{},
				Required:   []string{},
			},
		}
	}
}

func WithProperty(
	name string,
	typ string,
	description string,
	enum []any,
) OptionTool {
	return func(tool *Tool) {
		tool.Function.Parameters.Properties = append(tool.Function.Parameters.Properties, Property{
			Name:        name,
			Type:        typ,
			Description: description,
			Enum:        enum,
		})
	}
}

func WithRequired(required ...string) OptionTool {
	return func(tool *Tool) {
		tool.Function.Parameters.Required = required
	}
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
	Model            string          `json:"model"`
	Messages         []*Message      `json:"messages"`
	Tools            []*Tool         `json:"tools"`
	ResponseFormat   *ResponseFormat `json:"response_format"`
	Temperature      float64         `json:"temperature"`
	TopP             float64         `json:"top_p"`
	TopK             float64         `json:"top_k"`
	FrequencyPenalty float64         `json:"frequency_penalty"`
	PresencePenalty  float64         `json:"presence_penalty"`
	MaxTokens        int             `json:"max_tokens"`
	Stream           bool            `json:"stream"`
}

type OptionParams func(*Params)

func NewParams(opts ...OptionParams) *Params {
	params := &Params{}

	for _, opt := range opts {
		opt(params)
	}

	return params
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

func WithModel(model string) OptionParams {
	return func(params *Params) { params.Model = model }
}

func WithMessages(messages ...*Message) OptionParams {
	return func(params *Params) {
		for _, message := range messages {
			params.Messages = append(params.Messages, message)
		}
	}
}

func WithTools(tools ...*Tool) OptionParams {
	return func(params *Params) {
		for _, tool := range tools {
			params.Tools = append(params.Tools, tool)
		}
	}
}

func WithResponseFormat(responseFormat *ResponseFormat) OptionParams {
	return func(params *Params) { params.ResponseFormat = responseFormat }
}

func WithTemperature(temperature float64) OptionParams {
	return func(params *Params) { params.Temperature = temperature }
}

func WithTopP(topP float64) OptionParams {
	return func(params *Params) { params.TopP = topP }
}

func WithTopK(topK float64) OptionParams {
	return func(params *Params) { params.TopK = topK }
}

func WithFrequencyPenalty(frequencyPenalty float64) OptionParams {
	return func(params *Params) { params.FrequencyPenalty = frequencyPenalty }
}

func WithPresencePenalty(presencePenalty float64) OptionParams {
	return func(params *Params) { params.PresencePenalty = presencePenalty }
}

func WithMaxTokens(maxTokens int) OptionParams {
	return func(params *Params) { params.MaxTokens = maxTokens }
}

func WithStream(stream bool) OptionParams {
	return func(params *Params) { params.Stream = stream }
}
