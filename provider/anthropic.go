package provider

import (
	"context"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/theapemachine/errnie"
)

type Anthropic struct {
	*BaseProvider
	client *sdk.Client
	model  string
	cancel context.CancelFunc
}

func NewAnthropic(apiKey string) *Anthropic {
	return &Anthropic{
		BaseProvider: NewBaseProvider(),
		client:       sdk.NewClient(option.WithAPIKey(apiKey)),
		model:        sdk.ModelClaude3_5SonnetLatest,
	}
}

/*
Name returns the name of the provider.
*/
func (anthropic *Anthropic) Name() string {
	return "anthropic (claude 3.5 sonnet)"
}

func (anthropic *Anthropic) Generate(ctx context.Context, params *LLMGenerationParams) (<-chan Event, error) {
	out := make(chan Event)
	ctx, cancel := context.WithCancel(ctx)
	anthropic.cancel = cancel

	go func() {
		defer close(out)
		defer cancel()

		// Send start event
		startEvent := NewEventData()
		startEvent.EventType = EventStart
		startEvent.Name = "anthropic_generation_start"
		out <- startEvent

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
					Name:        sdk.F(tool.Name()),
					Description: sdk.F(tool.Description()),
					InputSchema: sdk.F(tool.GenerateSchema()),
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
			select {
			case <-ctx.Done():
				return
			default:
				event := stream.Current()

				switch event := event.AsUnion().(type) {
				case sdk.ContentBlockStartEvent:
					if event.ContentBlock.Name != "" {
						startEvent := NewEventData()
						startEvent.EventType = EventStart
						startEvent.Name = "anthropic_block_start"
						startEvent.Text = event.ContentBlock.Name
						out <- startEvent
					}
				case sdk.ContentBlockDeltaEvent:
					chunkEvent := NewEventData()
					chunkEvent.EventType = EventChunk
					chunkEvent.Name = "anthropic_chunk"
					chunkEvent.Text = event.Delta.Text
					chunkEvent.PartialJSON = event.Delta.PartialJSON
					out <- chunkEvent
				case sdk.ContentBlockStopEvent:
					doneEvent := NewEventData()
					doneEvent.EventType = EventDone
					doneEvent.Name = "anthropic_block_complete"
					doneEvent.Text = "\n"
					out <- doneEvent
				case sdk.MessageStopEvent:
					doneEvent := NewEventData()
					doneEvent.EventType = EventDone
					doneEvent.Name = "anthropic_generation_complete"
					doneEvent.Text = "\n"
					out <- doneEvent
				}
			}
		}

		if err := stream.Err(); err != nil {
			errnie.Error(err)
			errnie.Log("DUMP %v", params)
			errEvent := NewEventData()
			errEvent.EventType = EventError
			errEvent.Name = "anthropic_error"
			errEvent.Error = err
			out <- errEvent
			return
		}
	}()

	return out, nil
}

func (anthropic *Anthropic) CancelGeneration(ctx context.Context) error {
	if anthropic.cancel != nil {
		anthropic.cancel()
	}
	return nil
}

// Version returns the provider version
func (a *Anthropic) Version() string {
	return "1.0.0"
}

// Initialize sets up the provider
func (a *Anthropic) Initialize(ctx context.Context) error {
	return nil
}

// PauseGeneration pauses the current generation
func (a *Anthropic) PauseGeneration() error {
	return nil
}

// ResumeGeneration resumes the current generation
func (a *Anthropic) ResumeGeneration() error {
	return nil
}

// GetCapabilities returns the provider capabilities
func (a *Anthropic) GetCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"streaming": true,
		"tools":     true,
	}
}

// SupportsFeature checks if a feature is supported
func (a *Anthropic) SupportsFeature(feature string) bool {
	caps := a.GetCapabilities()
	supported, ok := caps[feature].(bool)
	return ok && supported
}

// ValidateConfig validates the provider configuration
func (a *Anthropic) ValidateConfig() error {
	return nil
}

// Cleanup performs any necessary cleanup
func (a *Anthropic) Cleanup(ctx context.Context) error {
	if a.client != nil {
		a.client = nil
	}
	return nil
}
