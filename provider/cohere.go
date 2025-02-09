package provider

import (
	"context"
	"errors"
	"io"

	"github.com/charmbracelet/log"
	sdk "github.com/cohere-ai/cohere-go/v2"
	client "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/davecgh/go-spew/spew"
	"github.com/spf13/viper"
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
		model:        viper.GetViper().GetString("models.cohere"),
	}
}

// Version returns the provider version
func (cohere *Cohere) Version() string {
	return "1.0.0"
}

// Initialize sets up the provider
func (cohere *Cohere) Initialize(ctx context.Context) error {
	return nil
}

// PauseGeneration pauses the current generation
func (cohere *Cohere) PauseGeneration() error {
	return nil
}

// ResumeGeneration resumes the current generation
func (cohere *Cohere) ResumeGeneration() error {
	return nil
}

// GetCapabilities returns the provider capabilities
func (cohere *Cohere) GetCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"streaming": true,
		"tools":     true,
	}
}

// SupportsFeature checks if a feature is supported
func (cohere *Cohere) SupportsFeature(feature string) bool {
	caps := cohere.GetCapabilities()
	supported, ok := caps[feature].(bool)
	return ok && supported
}

// ValidateConfig validates the provider configuration
func (cohere *Cohere) ValidateConfig() error {
	return nil
}

// Cleanup performs any necessary cleanup
func (cohere *Cohere) Cleanup(ctx context.Context) error {
	if cohere.client != nil {
		cohere.client = nil
	}
	return nil
}

// CancelGeneration cancels any ongoing generation
func (cohere *Cohere) CancelGeneration(ctx context.Context) error {
	if cohere.cancel != nil {
		cohere.cancel()
	}
	return nil
}

func (cohere *Cohere) Name() string {
	return "cohere (command-r)"
}

func (cohere *Cohere) Generate(params *LLMGenerationParams) <-chan *Event {
	errnie.Info("selected provider", "provider", "cohere", "model", cohere.model)

	out := make(chan *Event)
	ctx, cancel := context.WithCancel(context.Background())
	cohere.cancel = cancel

	go func() {
		defer close(out)
		defer cancel()

		// Send start event
		startEvent := NewEvent("generate:start", EventStart, "cohere:"+cohere.model, "", nil)
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
				if msg.Content == "" {
					continue
				}

				history[i] = cohere.convertMessages(msg)
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
			log.Error("Error streaming Cohere response", "error", err)
			spew.Dump(params)
			errEvent := NewEvent("generate:error", EventError, err.Error(), "", nil)
			out <- errEvent
			return
		}

		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				doneEvent := NewEvent("generate:stop", EventStop, "", "", nil)
				out <- doneEvent
				return
			}

			if err != nil {
				log.Error("Error streaming Cohere response", "error", err)
				errEvent := NewEvent("generate:error", EventError, err.Error(), "", nil)
				out <- errEvent
				return
			}

			if event := resp.StreamStart; event != nil {
				startEvent := NewEvent("generate:start", EventStart, "cohere:"+cohere.model, "", nil)
				out <- startEvent
				continue
			}

			if event := resp.TextGeneration; event != nil {
				chunkEvent := NewEvent("generate:contentblock:delta", EventChunk, event.Text, "", nil)
				out <- chunkEvent
				continue
			}

			if event := resp.ToolCallsChunk; event != nil {
				toolEvent := NewEvent("generate:toolcall", EventFunction, "Tool called", "", nil)
				out <- toolEvent
				continue
			}

			if event := resp.ToolCallsGeneration; event != nil {
				toolEvent := NewEvent("generate:toolcall", EventFunction, "Tool called", "", nil)
				out <- toolEvent
				continue
			}

			if event := resp.StreamEnd; event != nil {
				doneEvent := NewEvent("generate:stop", EventStop, "", "", nil)
				if event.Response != nil {
					doneEvent.Text = event.Response.Text
				}
				out <- doneEvent
				continue
			}
		}
	}()

	return out
}

func (cohere *Cohere) convertMessages(msg *Message) *sdk.Message {
	switch msg.Role {
	case RoleSystem:
		return &sdk.Message{
			Role: string(msg.Role),
			System: &sdk.ChatMessage{
				Message: msg.Content,
			},
		}
	case RoleUser:
		return &sdk.Message{
			Role: string(msg.Role),
			User: &sdk.ChatMessage{
				Message: msg.Content,
			},
		}
	case RoleAssistant:
		return &sdk.Message{
			Role: string(msg.Role),
			Chatbot: &sdk.ChatMessage{
				Message: msg.Content,
			},
		}
	}

	return nil
}
