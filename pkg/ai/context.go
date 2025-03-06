package ai

import (
	"github.com/theapemachine/caramba/pkg/core"
)

type Context struct {
	Model            string          `json:"model"`
	Messages         []*core.Message `json:"messages"`
	Tools            []*core.Tool    `json:"tools"`
	Process          *core.Process   `json:"process"`
	Temperature      float64         `json:"temperature"`
	TopP             float64         `json:"top_p"`
	TopK             int             `json:"top_k"`
	PresencePenalty  float64         `json:"presence_penalty"`
	FrequencyPenalty float64         `json:"frequency_penalty"`
	MaxTokens        int             `json:"max_tokens"`
	StopSequences    []string        `json:"stop_sequences"`
	Stream           bool            `json:"stream"`
}

func NewContext() *Context {
	return &Context{
		Model:       "gpt-4o-mini",
		Messages:    []*core.Message{},
		Tools:       []*core.Tool{},
		Temperature: 0.5,
		TopP:        1.0,
		Stream:      true,
	}
}
