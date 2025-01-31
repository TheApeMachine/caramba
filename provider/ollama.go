package provider

import (
	"context"
	"errors"
	"net/http"
	"net/url"

	sdk "github.com/ollama/ollama/api"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

type Ollama struct {
	*BaseProvider
	client *sdk.Client
	model  string
	cancel context.CancelFunc
}

func NewOllama(host string) *Ollama {
	hostURL, _ := url.Parse(host)
	return &Ollama{
		BaseProvider: NewBaseProvider(),
		client:       sdk.NewClient(hostURL, http.DefaultClient),
		model:        "llama3.2:3b",
	}
}

func (ollama *Ollama) Name() string {
	return "ollama (llama3.2:3b)"
}

func (ollama *Ollama) Generate(ctx context.Context, params *LLMGenerationParams) <-chan *Event {
	out := make(chan *Event)
	ctx, cancel := context.WithCancel(ctx)
	ollama.cancel = cancel

	go func() {
		defer close(out)
		defer cancel()

		// Send start event
		startEvent := NewEvent("generate:start", EventStart, "ollama:llama3.2:3b", "", nil)
		out <- startEvent

		// Convert our tools to Ollama format only if tools exist
		var tools []sdk.Tool
		if len(params.Tools) > 0 {
			tools = make([]sdk.Tool, len(params.Tools))
			for i, tool := range params.Tools {
				tools[i] = sdk.Tool{
					Type: "function",
					Function: sdk.ToolFunction{
						Name:        tool.Name(),
						Description: tool.Description(),
						Parameters:  tool.GenerateSchema().(sdk.ToolFunction).Parameters,
					},
				}
			}
		}

		// Convert our messages to Ollama format only if messages exist
		var messages []sdk.Message
		if len(params.Thread.Messages) > 0 {
			messages = make([]sdk.Message, len(params.Thread.Messages))
			for i, msg := range params.Thread.Messages {
				if msg.Content != "" {
					messages[i] = sdk.Message{
						Role:    string(msg.Role),
						Content: msg.Content,
					}
				}
			}
		}

		// Only proceed if we have messages
		if len(messages) == 0 {
			err := errors.New("no valid messages to process")
			errnie.Error(err)
			errEvent := NewEvent("generate:error", EventError, err.Error(), "", nil)
			out <- errEvent
			return
		}

		// Build request with only non-empty fields
		request := &sdk.ChatRequest{
			Model:    ollama.model,
			Messages: messages,
			Stream:   utils.BoolPtr(true),
		}

		// Only add tools if we have any
		if len(tools) > 0 {
			request.Tools = sdk.Tools(tools)
		}

		done := make(chan struct{})
		var streamErr error

		go func() {
			defer close(done)
			streamErr = ollama.client.Chat(ctx, request, func(resp sdk.ChatResponse) error {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
					if resp.Message.Content != "" {
						chunkEvent := NewEvent("generate:contentblock:delta", EventChunk, resp.Message.Content, "", nil)
						out <- chunkEvent
					}
					return nil
				}
			})
		}()

		select {
		case <-ctx.Done():
			return
		case <-done:
			if streamErr != nil {
				errnie.Error(streamErr)
				errEvent := NewEvent("generate:error", EventError, streamErr.Error(), "", nil)
				out <- errEvent
				return
			}
		}

		// Send done event
		doneEvent := NewEvent("generate:stop", EventStop, "\n", "", nil)
		out <- doneEvent
	}()

	return out
}

func (ollama *Ollama) CancelGeneration(ctx context.Context) error {
	if ollama.cancel != nil {
		ollama.cancel()
	}
	return nil
}

// Version returns the provider version
func (o *Ollama) Version() string {
	return "1.0.0"
}

// Initialize sets up the provider
func (o *Ollama) Initialize(ctx context.Context) error {
	return nil
}

// PauseGeneration pauses the current generation
func (o *Ollama) PauseGeneration() error {
	return nil
}

// ResumeGeneration resumes the current generation
func (o *Ollama) ResumeGeneration() error {
	return nil
}

// GetCapabilities returns the provider capabilities
func (o *Ollama) GetCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"streaming": true,
		"tools":     true,
	}
}

// SupportsFeature checks if a feature is supported
func (o *Ollama) SupportsFeature(feature string) bool {
	caps := o.GetCapabilities()
	supported, ok := caps[feature].(bool)
	return ok && supported
}

// ValidateConfig validates the provider configuration
func (o *Ollama) ValidateConfig() error {
	return nil
}

// Cleanup performs any necessary cleanup
func (o *Ollama) Cleanup(ctx context.Context) error {
	if o.client != nil {
		o.client = nil
	}
	return nil
}
