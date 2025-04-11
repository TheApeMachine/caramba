package provider

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/ollama/ollama/api"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
OllamaProvider implements an LLM provider that connects to Ollama's API.
It supports regular chat completions and streaming responses.
*/
type OllamaProvider struct {
	client   *api.Client
	endpoint string
	ctx      context.Context
	cancel   context.CancelFunc
}

/*
NewOllamaProvider creates a new Ollama provider with the given host endpoint.
If host is empty, it will try to read from configuration.
*/
func NewOllamaProvider(opts ...OllamaProviderOption) *OllamaProvider {
	errnie.Debug("provider.NewOllamaProvider")

	endpoint := viper.GetViper().GetString("endpoints.ollama")
	ctx, cancel := context.WithCancel(context.Background())

	client, err := api.ClientFromEnvironment()
	if errnie.Error(err) != nil {
		cancel()
		return nil
	}

	prvdr := &OllamaProvider{
		client:   client,
		endpoint: endpoint,
		ctx:      ctx,
		cancel:   cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *OllamaProvider) ID() string {
	return "ollama"
}

type OllamaProviderOption func(*OllamaProvider)

func WithOllamaEndpoint(endpoint string) OllamaProviderOption {
	return func(prvdr *OllamaProvider) {
		prvdr.endpoint = endpoint
	}
}

func (prvdr *OllamaProvider) Generate(
	params ProviderParams,
) (ProviderEvent, error) {
	errnie.Info("provider.Generate", "supplier", "ollama")

	composed := &api.ChatRequest{
		Model: params.Model,
		Options: map[string]interface{}{
			"temperature": params.Temperature,
			"top_p":       params.TopP,
			"max_tokens":  params.MaxTokens,
		},
	}

	var err error

	if err = prvdr.buildMessages(composed, params.Messages); err != nil {
		return ProviderEvent{}, err
	}

	// Get tools from the artifact metadata
	if err = prvdr.buildTools(composed, params.Tools); err != nil {
		return ProviderEvent{}, err
	}

	if params.ResponseFormat != (ResponseFormat{}) {
		if err = prvdr.buildResponseFormat(composed, params.ResponseFormat); err != nil {
			return ProviderEvent{}, err
		}
	}

	if params.Stream {
		return prvdr.handleStreamingRequest(composed)
	}

	return prvdr.handleSingleRequest(composed)
}

func (prvdr *OllamaProvider) Name() string {
	return "ollama"
}

func (prvdr *OllamaProvider) handleSingleRequest(
	params *api.ChatRequest,
) (ProviderEvent, error) {
	errnie.Debug("provider.handleSingleRequest")

	var responseContent string

	err := prvdr.client.Chat(
		prvdr.ctx, params, func(response api.ChatResponse) error {
			responseContent = response.Message.Content
			return nil
		},
	)

	if errnie.Error(err) != nil {
		return ProviderEvent{}, errnie.Error(err)
	}

	// Return provider event with message
	return ProviderEvent{
		Message: Message{
			Role:    "assistant",
			Name:    params.Model,
			Content: responseContent,
		},
	}, nil
}

func (prvdr *OllamaProvider) handleStreamingRequest(
	params *api.ChatRequest,
) (ProviderEvent, error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream := true
	params.Stream = &stream

	var lastContent string
	err := prvdr.client.Chat(prvdr.ctx, params, func(response api.ChatResponse) error {
		if response.Message.Content != "" {
			lastContent = response.Message.Content
			return nil
		}
		return nil
	})

	if errnie.Error(err) != nil {
		return ProviderEvent{}, errnie.Error(err)
	}

	// Return the latest content
	return ProviderEvent{
		Message: Message{
			Role:    "assistant",
			Name:    params.Model,
			Content: lastContent,
		},
	}, nil
}

func (prvdr *OllamaProvider) buildMessages(
	chatParams *api.ChatRequest,
	messages []Message,
) (err error) {
	errnie.Debug("provider.buildMessages")

	messageList := make([]api.Message, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			messageList = append(messageList, api.Message{
				Role:    "system",
				Content: msg.Content,
			})
		case "user":
			messageList = append(messageList, api.Message{
				Role:    "user",
				Content: msg.Content,
			})
		case "assistant":
			// For Ollama, we currently don't handle tool calls in history
			// But we can still add the assistant message
			messageList = append(messageList, api.Message{
				Role:    "assistant",
				Content: msg.Content,
			})
		case "tool":
			// Ollama doesn't have a tool message type, so we'll add it as a user message
			// with the content indicating it's a tool result
			messageList = append(messageList, api.Message{
				Role:    "user",
				Content: fmt.Sprintf("[Tool Result from %s: %s]", msg.ID, msg.Content),
			})
		default:
			errnie.Error("unknown message role", "role", msg.Role)
		}
	}

	chatParams.Messages = messageList

	return nil
}

func (prvdr *OllamaProvider) buildTools(
	chatParams *api.ChatRequest,
	tools []mcp.Tool,
) (err error) {
	errnie.Debug("provider.buildTools")

	if len(tools) == 0 {
		return
	}

	toolList := make([]api.Tool, 0, len(tools))

	for _, tool := range tools {
		toolList = append(toolList, api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters: struct {
					Type       string   `json:"type"`
					Required   []string `json:"required"`
					Properties map[string]struct {
						Type        string   `json:"type"`
						Description string   `json:"description"`
						Enum        []string `json:"enum,omitempty"`
					} `json:"properties"`
				}{
					Type:     "object",
					Required: []string{}, // Ollama doesn't seem to support required fields the same way
					Properties: map[string]struct {
						Type        string   `json:"type"`
						Description string   `json:"description"`
						Enum        []string `json:"enum,omitempty"`
					}{
						"input": {
							Type:        "string",
							Description: "The input to the function",
						},
					},
				},
			},
		})
	}

	if len(toolList) > 0 {
		chatParams.Tools = toolList
	}

	return nil
}

func (prvdr *OllamaProvider) buildResponseFormat(
	chatParams *api.ChatRequest,
	format ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	// Add format instructions as a system message since Ollama doesn't support direct format control
	if format.Name != "" || format.Description != "" || format.Schema != nil {
		schemaStr := ""
		if format.Schema != nil {
			switch s := format.Schema.(type) {
			case string:
				schemaStr = s
			default:
				j, _ := json.Marshal(format.Schema)
				schemaStr = string(j)
			}
		}

		formatMsg := api.Message{
			Role: "system",
			Content: "Please format your response according to the specified schema: " +
				format.Name + ". " + format.Description + "\nSchema: " + schemaStr,
		}
		chatParams.Messages = append(chatParams.Messages, formatMsg)
	}

	return nil
}
