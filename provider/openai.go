package provider

import (
	"context"
	"errors"
	"time"

	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type OpenAI struct {
	client *sdk.Client
	model  string
}

func NewOpenAI(apiKey, model string) *OpenAI {
	return &OpenAI{
		client: sdk.NewClient(
			option.WithAPIKey(apiKey),
		),
		model: model,
	}
}

func (o *OpenAI) Generate(params GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		messages := o.convertToOpenAIMessages(params.Messages)

		if len(messages) == 0 {
			out <- Event{
				Type:  EventError,
				Error: errors.New("no valid messages to send"),
			}
			return
		}

		stream := o.client.Chat.Completions.NewStreaming(context.Background(), sdk.ChatCompletionNewParams{
			Messages:    sdk.F(messages),
			Model:       sdk.F(o.model),
			Temperature: sdk.F(params.Temperature),
		})

		for stream.Next() {
			evt := stream.Current()
			if len(evt.Choices) > 0 && evt.Choices[0].Delta.Content != "" {
				out <- Event{
					Sequence: time.Now().UnixNano(),
					Type:     EventToken,
					Content:  evt.Choices[0].Delta.Content,
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

func (o *OpenAI) convertToOpenAIMessages(messages []Message) []sdk.ChatCompletionMessageParamUnion {
	result := make([]sdk.ChatCompletionMessageParamUnion, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case "user":
			result = append(result, sdk.UserMessage(msg.Content))
		case "assistant":
			result = append(result, sdk.AssistantMessage(msg.Content))
		case "system":
			result = append(result, sdk.SystemMessage(msg.Content))
		}
	}

	return result
}
