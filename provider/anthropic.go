package provider

import (
	"context"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/davecgh/go-spew/spew"
	"github.com/theapemachine/errnie"
)

type Anthropic struct {
	client *sdk.Client
	model  string
}

func NewAnthropic(apiKey string) *Anthropic {
	return &Anthropic{
		client: sdk.NewClient(option.WithAPIKey(apiKey)),
		model:  sdk.ModelClaude3_5SonnetLatest,
	}
}

func (anthropic *Anthropic) Generate(ctx context.Context, params *GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		// Initialize base params
		messageParams := sdk.MessageNewParams{
			Model:     sdk.F(anthropic.model),
			MaxTokens: sdk.F(params.MaxTokens),
		}

		// Only add tools if they exist
		if len(params.Tools) > 0 {
			tools := make([]sdk.ToolParam, len(params.Tools))
			for i, tool := range params.Tools {
				tools[i] = sdk.ToolParam{
					Name:        sdk.F(tool.Name),
					Description: sdk.F(tool.Description),
					InputSchema: sdk.F(tool.Schema),
				}
			}
			messageParams.Tools = sdk.F(tools)
		}

		// Process messages
		thread := make([]sdk.MessageParam, 0)
		systemMessage := []sdk.TextBlockParam{}

		for _, message := range params.Thread.Messages {
			switch message.Role {
			case RoleSystem:
				systemMessage = append(systemMessage, sdk.NewTextBlock(message.Content))
			case RoleUser:
				thread = append(thread, sdk.NewUserMessage(sdk.NewTextBlock(message.Content)))
			case RoleAssistant:
				thread = append(thread, sdk.NewAssistantMessage(sdk.NewTextBlock(message.Content)))
			}
		}

		// Only add system message if we have one
		if len(systemMessage) > 0 {
			messageParams.System = sdk.F(systemMessage)
		}

		// Only add messages if we have any
		if len(thread) > 0 {
			messageParams.Messages = sdk.F(thread)
		}

		stream := anthropic.client.Messages.NewStreaming(ctx, messageParams)

		for stream.Next() {
			event := stream.Current()

			switch event := event.AsUnion().(type) {
			case sdk.ContentBlockStartEvent:
				if event.ContentBlock.Name != "" {
					out <- Event{Type: EventStart, Text: event.ContentBlock.Name}
				}
			case sdk.ContentBlockDeltaEvent:
				out <- Event{Type: EventChunk, Text: event.Delta.Text, PartialJSON: event.Delta.PartialJSON}
			case sdk.ContentBlockStopEvent:
				out <- Event{Type: EventDone, Text: "\n"}
			case sdk.MessageStopEvent:
				out <- Event{Type: EventDone, Text: "\n"}
			}
		}

		if err := stream.Err(); err != nil {
			errnie.Error(err)
			spew.Dump(params)
			out <- Event{Type: EventError, Error: err}
			return
		}
	}()

	return out
}
