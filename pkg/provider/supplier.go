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

// PendingToolCall holds details needed to execute a tool and correlate its result.
type PendingToolCall struct {
	ID      string              `json:"id"`      // ID from the LLM's tool_call request
	Request mcp.CallToolRequest `json:"request"` // Parsed request details (name, args)
}

type ProviderEvent struct {
	Message   Message           `json:"message"`
	ToolCalls []PendingToolCall `json:"tool_calls"` // Changed from []mcp.CallToolRequest
}

// Interface for LLM providers
type ProviderType interface {
	// Generate initiates the LLM call. For non-streaming, it returns a channel
	// with one event. For streaming, it returns a channel that emits events as they arrive.
	Generate(params ProviderParams) (<-chan ProviderEvent, error)
	Name() string
}

type Message struct {
	ID          string                `json:"id"` // Can be used for message ID, OR ToolCallID for 'tool' role
	Role        string                `json:"role"`
	Name        string                `json:"name"`         // Optional: Model name for assistant, Tool name for tool result
	Content     string                `json:"content"`      // Text content OR result content for 'tool' role
	ToolCalls   []mcp.CallToolRequest `json:"tool_calls"`   // Tool calls requested *by* this assistant message
	ToolResults []mcp.CallToolResult  `json:"tool_results"` // Results *provided* in response to tool calls (maybe remove if ID handles correlation?)
}

type ResponseFormat struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Schema      any    `json:"schema"`
	Strict      bool   `json:"strict"`
}

type Embedder interface {
	Generate(document string) (embeddings []float32)
}
