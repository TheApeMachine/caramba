package provider

import (
	"context"
	"errors"
	"fmt"
	"io"

	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/spf13/viper"
	"github.com/theapemachine/errnie"
)

type DeepSeek struct {
	*BaseProvider
	client *deepseek.Client
	model  string
	cancel context.CancelFunc
}

func NewDeepSeek(apiKey string) *DeepSeek {
	return &DeepSeek{
		BaseProvider: NewBaseProvider(),
		client:       deepseek.NewClient(apiKey),
		model:        viper.GetViper().GetString("models.deepseek"),
	}
}

func (d *DeepSeek) Name() string {
	return fmt.Sprintf("deepseek (%s)", d.model)
}

func (d *DeepSeek) Generate(params *LLMGenerationParams) <-chan *Event {
	out := make(chan *Event)
	ctx, cancel := context.WithCancel(context.Background())
	d.cancel = cancel

	go func() {
		defer close(out)
		defer cancel()

		// Send start event
		startEvent := NewEvent("generate:start", EventStart, "deepseek:"+d.model, "", nil)
		out <- startEvent

		messages := d.convertMessages(params)

		// Only proceed if we have messages
		if len(messages) == 0 {
			err := errors.New("no valid messages to process")
			errnie.Error(err)
			errEvent := NewEvent("generate:error", EventError, err.Error(), "", nil)
			out <- errEvent
			return
		}

		request := &deepseek.StreamChatCompletionRequest{
			Model:    d.model,
			Messages: messages,
			Stream:   true,
		}

		// Add optional parameters
		if params.MaxTokens > 0 {
			request.MaxTokens = int(params.MaxTokens)
		}
		if params.Temperature > 0 {
			request.Temperature = float32(params.Temperature)
		}

		stream, err := d.client.CreateChatCompletionStream(ctx, request)
		if err != nil {
			errnie.Error(err)
			errEvent := NewEvent("generate:error", EventError, err.Error(), "", nil)
			out <- errEvent
			return
		}
		defer stream.Close()

		for {
			select {
			case <-ctx.Done():
				return
			default:
				response, err := stream.Recv()
				if err != nil {
					if errors.Is(err, io.EOF) {
						// Send done event
						doneEvent := NewEvent("generate:stop", EventStop, "\n", "", nil)
						out <- doneEvent
						return
					}
					errnie.Error(err)
					errEvent := NewEvent("generate:error", EventError, err.Error(), "", nil)
					out <- errEvent
					return
				}

				for _, choice := range response.Choices {
					if choice.Delta.Content != "" {
						chunkEvent := NewEvent("generate:contentblock:delta", EventChunk, choice.Delta.Content, "", nil)
						out <- chunkEvent
					}
				}
			}
		}
	}()

	return out
}

func (d *DeepSeek) CancelGeneration(ctx context.Context) error {
	if d.cancel != nil {
		d.cancel()
	}
	return nil
}

func (*DeepSeek) convertMessages(params *LLMGenerationParams) []deepseek.ChatCompletionMessage {
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

// Version returns the provider version
func (d *DeepSeek) Version() string {
	return "1.0.0"
}

// Initialize sets up the provider
func (d *DeepSeek) Initialize(ctx context.Context) error {
	return nil
}

// PauseGeneration pauses the current generation
func (d *DeepSeek) PauseGeneration() error {
	return nil
}

// ResumeGeneration resumes the current generation
func (d *DeepSeek) ResumeGeneration() error {
	return nil
}

// GetCapabilities returns the provider capabilities
func (d *DeepSeek) GetCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"streaming": true,
		"tools":     true,
	}
}

// SupportsFeature checks if a feature is supported
func (d *DeepSeek) SupportsFeature(feature string) bool {
	caps := d.GetCapabilities()
	supported, ok := caps[feature].(bool)
	return ok && supported
}

// ValidateConfig validates the provider configuration
func (d *DeepSeek) ValidateConfig() error {
	return nil
}

// Cleanup performs any necessary cleanup
func (d *DeepSeek) Cleanup(ctx context.Context) error {
	if d.client != nil {
		d.client = nil
	}
	return nil
}
