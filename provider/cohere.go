package provider

import (
	"context"
	"errors"
	"io"

	sdk "github.com/cohere-ai/cohere-go/v2"
	client "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/theapemachine/errnie"
)

type Cohere struct {
	client *client.Client
	model  string
}

func NewCohere(apiKey string) *Cohere {
	client := client.NewClient(client.WithToken(apiKey))
	return &Cohere{
		client: client,
		model:  "command-r",
	}
}

func (cohere *Cohere) Name() string {
	return "cohere (command-r)"
}

func (cohere *Cohere) Generate(ctx context.Context, params *GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		// Only add tools if they exist
		var tools []*sdk.Tool
		if len(params.Tools) > 0 {
			tools = make([]*sdk.Tool, len(params.Tools))
			for i, tool := range params.Tools {
				tools[i] = &sdk.Tool{
					Name:                 tool.Name,
					Description:          tool.Description,
					ParameterDefinitions: tool.Schema.(map[string]*sdk.ToolParameterDefinitionsValue),
				}
			}
		}

		maxTokens := int(params.MaxTokens)

		// Only add messages if we have any
		var history []*sdk.Message
		if len(params.Thread.Messages) > 0 {
			history = make([]*sdk.Message, len(params.Thread.Messages))
			for i, msg := range params.Thread.Messages {
				history[i] = &sdk.Message{
					Role: string(msg.Role),
					User: &sdk.ChatMessage{
						Message: msg.Content,
					},
				}
			}
		}

		// Build request with only non-empty fields
		request := &sdk.ChatStreamRequest{
			Model:     &cohere.model,
			MaxTokens: &maxTokens,
		}

		// Only add message if we have messages
		if len(params.Thread.Messages) > 0 {
			request.Message = params.Thread.Messages[len(params.Thread.Messages)-1].Content
		}

		// Only add tools if we have any
		if len(tools) > 0 {
			request.Tools = tools
		}

		// Only add history if we have any
		if len(history) > 0 {
			request.ChatHistory = history
		}

		stream, err := cohere.client.ChatStream(ctx, request)
		if err != nil {
			errnie.Error(err)
			out <- Event{Type: EventError, Error: err}
			return
		}

		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				out <- Event{Type: EventDone, Text: "\n"}
				return
			}

			if err != nil {
				errnie.Error(err)
				out <- Event{Type: EventError, Error: err}
				return
			}

			if event := resp.StreamStart; event != nil {
				out <- Event{Type: EventStart, Text: event.String()}
			}

			if event := resp.TextGeneration; event != nil {
				out <- Event{Type: EventChunk, Text: event.String()}
			}

			if event := resp.ToolCallsChunk; event != nil {
				out <- Event{Type: EventToolCall, Text: event.String()}
			}

			if event := resp.ToolCallsGeneration; event != nil {
				out <- Event{Type: EventToolCall, Text: event.String()}
			}

			if event := resp.StreamEnd; event != nil {
				out <- Event{Type: EventDone, Text: event.String()}
			}
		}
	}()

	return out
}
