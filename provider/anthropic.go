package provider

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/charmbracelet/log"
)

type Anthropic struct {
	client    *anthropic.Client
	model     string
	maxTokens int64
	system    string
}

func NewAnthropic(apiKey string, model string) *Anthropic {
	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
		option.WithHeader("x-api-key", apiKey),
	)
	return &Anthropic{
		client:    client,
		model:     model,
		maxTokens: 4096,
	}
}

func (a *Anthropic) Configure(config map[string]interface{}) {
	if systemMsg, ok := config["system_message"].(string); ok {
		a.system = systemMsg
	}
}

func (a *Anthropic) Generate(params GenerationParams) <-chan Event {
	log.Info("generating with", "model", a.model)
	out := make(chan Event)

	go func() {
		defer close(out)

		messages := a.convertToAnthropicMessages(params.Messages)

		stream := a.client.Messages.NewStreaming(context.Background(), anthropic.MessageNewParams{
			Model:     anthropic.F(a.model),
			MaxTokens: anthropic.F(a.maxTokens),
			Messages:  anthropic.F(messages),
		})

		for stream.Next() {
			event := stream.Current()
			if delta, ok := event.Delta.(anthropic.ContentBlockDeltaEventDelta); ok {
				if delta.Text != "" {
					out <- Event{
						Type:    EventToken,
						Content: delta.Text,
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			out <- Event{
				Type:  EventError,
				Error: err,
			}
			return
		}

		out <- Event{Type: EventDone, Content: "\n"}
	}()

	return out
}

func (a *Anthropic) convertToAnthropicMessages(messages []Message) []anthropic.MessageParam {
	result := make([]anthropic.MessageParam, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case "user":
			result = append(result, anthropic.NewUserMessage(anthropic.NewTextBlock(msg.Content)))
		case "assistant":
			result = append(result, anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.Content)))
		}
	}

	return result
}
