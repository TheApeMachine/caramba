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
	*BaseProvider
	client *client.Client
	model  string
	cancel context.CancelFunc
}

func NewCohere(apiKey string) *Cohere {
	return &Cohere{
		BaseProvider: NewBaseProvider(),
		client:       client.NewClient(client.WithToken(apiKey)),
		model:        "command-r",
	}
}

// Version returns the provider version
func (c *Cohere) Version() string {
	return "1.0.0"
}

// Initialize sets up the provider
func (c *Cohere) Initialize(ctx context.Context) error {
	return nil
}

// PauseGeneration pauses the current generation
func (c *Cohere) PauseGeneration() error {
	return nil
}

// ResumeGeneration resumes the current generation
func (c *Cohere) ResumeGeneration() error {
	return nil
}

// GetCapabilities returns the provider capabilities
func (c *Cohere) GetCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"streaming": true,
		"tools":     true,
	}
}

// SupportsFeature checks if a feature is supported
func (c *Cohere) SupportsFeature(feature string) bool {
	caps := c.GetCapabilities()
	supported, ok := caps[feature].(bool)
	return ok && supported
}

// ValidateConfig validates the provider configuration
func (c *Cohere) ValidateConfig() error {
	return nil
}

// Cleanup performs any necessary cleanup
func (c *Cohere) Cleanup(ctx context.Context) error {
	if c.client != nil {
		c.client = nil
	}
	return nil
}

// CancelGeneration cancels any ongoing generation
func (c *Cohere) CancelGeneration(ctx context.Context) error {
	if c.cancel != nil {
		c.cancel()
	}
	return nil
}

func (cohere *Cohere) Name() string {
	return "cohere (command-r)"
}

func (cohere *Cohere) Generate(ctx context.Context, params *LLMGenerationParams) <-chan Event {
	out := make(chan Event)
	ctx, cancel := context.WithCancel(ctx)
	cohere.cancel = cancel

	go func() {
		defer close(out)
		defer cancel()

		// Send start event
		startEvent := NewEventData()
		startEvent.EventType = EventStart
		startEvent.Name = "cohere_generation_start"
		out <- startEvent

		// Only add tools if they exist
		var tools []*sdk.Tool
		if len(params.Tools) > 0 {
			tools = make([]*sdk.Tool, len(params.Tools))
			for i, tool := range params.Tools {
				tools[i] = &sdk.Tool{
					Name:                 tool.Name(),
					Description:          tool.Description(),
					ParameterDefinitions: tool.GenerateSchema().(map[string]*sdk.ToolParameterDefinitionsValue),
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
			errEvent := NewEventData()
			errEvent.EventType = EventError
			errEvent.Error = err
			errEvent.Name = "cohere_error"
			out <- errEvent
			return
		}

		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				doneEvent := NewEventData()
				doneEvent.EventType = EventDone
				doneEvent.Name = "cohere_generation_complete"
				doneEvent.Text = "\n"
				out <- doneEvent
				return
			}

			if err != nil {
				errnie.Error(err)
				errEvent := NewEventData()
				errEvent.EventType = EventError
				errEvent.Error = err
				errEvent.Name = "cohere_error"
				out <- errEvent
				return
			}

			if event := resp.StreamStart; event != nil {
				startEvent := NewEventData()
				startEvent.EventType = EventStart
				startEvent.Name = "cohere_stream_start"
				out <- startEvent
			}

			if event := resp.TextGeneration; event != nil {
				chunkEvent := NewEventData()
				chunkEvent.EventType = EventChunk
				chunkEvent.Name = "cohere_chunk"
				chunkEvent.Text = event.Text
				out <- chunkEvent
			}

			if event := resp.ToolCallsChunk; event != nil {
				toolEvent := NewEventData()
				toolEvent.EventType = EventToolCall
				toolEvent.Name = "cohere_tool_call"
				toolEvent.Text = "Tool called"
				out <- toolEvent
			}

			if event := resp.ToolCallsGeneration; event != nil {
				toolEvent := NewEventData()
				toolEvent.EventType = EventToolCall
				toolEvent.Name = "cohere_tool_call"
				toolEvent.Text = "Tool called"
				out <- toolEvent
			}

			if event := resp.StreamEnd; event != nil {
				doneEvent := NewEventData()
				doneEvent.EventType = EventDone
				doneEvent.Name = "cohere_stream_end"
				if event.Response != nil {
					doneEvent.Text = event.Response.Text
				}
				out <- doneEvent
			}
		}
	}()

	return out
}
