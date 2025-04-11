package sampling

import (
	"context"
	"time"
)

// ModelPreferences represents preferences for model behavior
type ModelPreferences struct {
	Temperature      float64  `json:"temperature"`
	MaxTokens        int      `json:"maxTokens"`
	TopP             float64  `json:"topP"`
	FrequencyPenalty float64  `json:"frequencyPenalty"`
	PresencePenalty  float64  `json:"presencePenalty"`
	Stop             []string `json:"stop,omitempty"`
}

// Message represents a message in a sampling conversation
type Message struct {
	ID        string      `json:"id"`
	Role      string      `json:"role"`
	Content   string      `json:"content"`
	CreatedAt time.Time   `json:"createdAt"`
	Metadata  interface{} `json:"metadata,omitempty"`
}

// Context represents additional context for sampling
type Context struct {
	Messages []Message   `json:"messages"`
	Files    []string    `json:"files,omitempty"`
	Data     interface{} `json:"data,omitempty"`
}

// SamplingOptions represents options for message creation
type SamplingOptions struct {
	ModelPreferences ModelPreferences `json:"modelPreferences"`
	Context          *Context         `json:"context,omitempty"`
	Stream           bool             `json:"stream"`
}

// SamplingResult represents the result of a sampling operation
type SamplingResult struct {
	Message  Message     `json:"message"`
	Usage    Usage       `json:"usage"`
	Duration float64     `json:"duration"`
	Metadata interface{} `json:"metadata,omitempty"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"promptTokens"`
	CompletionTokens int `json:"completionTokens"`
	TotalTokens      int `json:"totalTokens"`
}

// SamplingManager defines the interface for sampling operations
type SamplingManager interface {
	// CreateMessage creates a new message using the provided options
	CreateMessage(ctx context.Context, content string, opts SamplingOptions) (*SamplingResult, error)

	// StreamMessage streams message tokens as they are generated
	StreamMessage(ctx context.Context, content string, opts SamplingOptions) (<-chan *SamplingResult, error)

	// GetModelPreferences returns the default model preferences
	GetModelPreferences(ctx context.Context) (*ModelPreferences, error)

	// UpdateModelPreferences updates the default model preferences
	UpdateModelPreferences(ctx context.Context, prefs ModelPreferences) error
}
