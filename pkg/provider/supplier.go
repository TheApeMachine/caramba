package provider

import "github.com/mark3labs/mcp-go/mcp"

type ProviderParams struct {
	Model            string
	Temperature      float64
	TopP             float64
	MaxTokens        int
	FrequencyPenalty float64
	PresencePenalty  float64
	Stream           bool
	ResponseFormat   ResponseFormat
	Messages         []Message
	Tools            []mcp.Tool
}

type ProviderEvent struct {
	Message   Message
	ToolCalls []mcp.CallToolRequest
}

type ProviderType interface {
	Generate()
}

type Message struct {
	ID          string                `json:"id"`
	Role        string                `json:"role"`
	Name        string                `json:"name"`
	Content     string                `json:"content"`
	ToolCalls   []mcp.CallToolRequest `json:"tool_calls"`
	ToolResults []mcp.CallToolResult  `json:"tool_results"`
}

type ResponseFormat struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Schema      any    `json:"schema"`
	Strict      bool   `json:"strict"`
}
