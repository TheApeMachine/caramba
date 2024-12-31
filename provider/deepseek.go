package provider

import (
	"context"
	"errors"
	"fmt"
	"io"

	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/theapemachine/errnie"
)

type DeepSeek struct {
	client *deepseek.Client
	model  string
}

func NewDeepSeek(apiKey string) *DeepSeek {
	return &DeepSeek{
		client: deepseek.NewClient(apiKey),
		model:  deepseek.DeepSeekChat,
	}
}

func (d *DeepSeek) Name() string {
	return fmt.Sprintf("deepseek (%s)", d.model)
}

func (d *DeepSeek) Generate(ctx context.Context, params *GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		messages := d.convertMessages(params)

		// Only proceed if we have messages
		if len(messages) == 0 {
			errnie.Error(errors.New("no valid messages to process"))
			out <- Event{Type: EventError, Error: errors.New("no valid messages to process")}
			return
		}

		request := &deepseek.StreamChatCompletionRequest{
			Model:    d.model,
			Messages: messages,
			Stream:   true,
		}

		// Add optional parameters
		if params.MaxTokens > 0 {
			request.MaxTokens = params.MaxTokens
		}
		if params.Temperature > 0 {
			request.Temperature = params.Temperature
		}

		stream, err := d.client.CreateChatCompletionStream(ctx, request)
		if err != nil {
			errnie.Error(err)
			out <- Event{Type: EventError, Error: err}
			return
		}
		defer stream.Close()

		for {
			response, err := stream.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				errnie.Error(err)
				out <- Event{Type: EventError, Error: err}
				return
			}

			for _, choice := range response.Choices {
				if choice.Delta.Content != "" {
					out <- Event{
						Type: EventChunk,
						Text: choice.Delta.Content,
					}
				}
			}
		}

		out <- Event{Type: EventDone, Text: "\n"}
	}()

	return out
}

func (*DeepSeek) convertMessages(params *GenerationParams) []deepseek.ChatCompletionMessage {
	var messages []deepseek.ChatCompletionMessage
	if len(params.Thread.Messages) > 0 {
		messages = make([]deepseek.ChatCompletionMessage, 0, len(params.Thread.Messages))
		for _, msg := range params.Thread.Messages {
			if msg.Content == "" {
				continue
			}

			message := deepseek.ChatCompletionMessage{
				Content: msg.Content,
			}

			switch msg.Role {
			case RoleSystem:
				message.Role = deepseek.ChatMessageRoleSystem
			case RoleUser:
				message.Role = deepseek.ChatMessageRoleUser
			case RoleAssistant:
				message.Role = deepseek.ChatMessageRoleAssistant
			}

			messages = append(messages, message)
		}
	}
	return messages
}
