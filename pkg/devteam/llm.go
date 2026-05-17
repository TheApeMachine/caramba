package devteam

import (
	"context"
	"strings"

	devcfg "github.com/theapemachine/caramba/pkg/config"
)

/*
ToolCall is the provider-neutral representation of a model's request to invoke
one of the sandbox tools.
*/
type ToolCall struct {
	ID    string
	Name  string
	Input map[string]any
}

/*
ToolDefinition is a provider-neutral JSON-Schema tool declaration.
*/
type ToolDefinition struct {
	Name        string
	Description string
	Parameters  map[string]any
}

/*
ChatMessage is a single turn in the conversation history.
Role is one of: "user", "assistant", "tool".
ToolCallID is only set for role="tool" (the call being answered).
ToolCalls is only set for role="assistant" when the model requested tool use.
*/
type ChatMessage struct {
	Role       string
	Content    string
	ToolCallID string
	ToolCalls  []ToolCall
}

/*
ChatRequest is a provider-neutral conversation payload.
*/
type ChatRequest struct {
	System    string
	Messages  []ChatMessage
	Tools     []ToolDefinition
	MaxTokens int
}

/*
ChatResponse is the assistant's reply for one turn.
*/
type ChatResponse struct {
	Content      string
	ToolCalls    []ToolCall
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
}

/*
Provider is the interface every LLM backend must satisfy.
A single Chat call maps one full conversation turn.
*/
type Provider interface {
	Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)
}

/*
NewProvider constructs the appropriate Provider from a ProviderConfig.
"anthropic" routes to AnthropicProvider; everything else goes through the
OpenAI provider, which selects the request shape from the configured endpoint.
*/
func NewProvider(cfg devcfg.ProviderConfig) Provider {
	switch strings.ToLower(cfg.Provider) {
	case "anthropic":
		return NewAnthropicProvider(cfg)
	default:
		return NewOpenAIProvider(cfg)
	}
}
