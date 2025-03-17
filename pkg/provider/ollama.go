package provider

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/spf13/viper"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
OllamaProvider implements an LLM provider that connects to Ollama's API.
It supports regular chat completions and streaming responses.
*/
type OllamaProvider struct {
	client *api.Client
	model  string
	buffer *stream.Buffer
	params *aiCtx.Artifact
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewOllamaProvider creates a new Ollama provider with the given host endpoint.
If host is empty, it will try to read from configuration.
*/
func NewOllamaProvider(
	host string,
	model string,
) *OllamaProvider {
	errnie.Debug("provider.NewOllamaProvider")

	if host == "" {
		host = viper.GetViper().GetString("endpoints.ollama")
	}

	if model == "" {
		model = "llama2" // Default model
	}

	ctx, cancel := context.WithCancel(context.Background())

	client, err := api.ClientFromEnvironment()
	if errnie.Error(err) != nil {
		return nil
	}

	prvdr := &OllamaProvider{
		client: client,
		model:  model,
		params: aiCtx.New(
			model,
			nil,
			nil,
			nil,
			0.7,
			1.0,
			0,
			0.0,
			0.0,
			2048,
			false,
		),
		ctx:    ctx,
		cancel: cancel,
	}

	prvdr.buffer = stream.NewBuffer(
		func(event *event.Artifact) error {
			errnie.Debug("provider.OllamaProvider.buffer.fn", "event", event)

			payload, err := event.Payload()
			if errnie.Error(err) != nil {
				return err
			}

			_, err = prvdr.params.Write(payload)
			if errnie.Error(err) != nil {
				return err
			}

			return nil
		},
	)

	return prvdr
}

/*
Read implements the io.Reader interface.
*/
func (prvdr *OllamaProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OllamaProvider.Read")
	return prvdr.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (prvdr *OllamaProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OllamaProvider.Write")

	n, err = prvdr.buffer.Write(p)
	if errnie.Error(err) != nil {
		return n, err
	}

	composed := &api.ChatRequest{
		Model: prvdr.model,
	}

	prvdr.buildMessages(prvdr.params, composed)
	prvdr.buildTools(prvdr.params, composed)
	prvdr.buildResponseFormat(prvdr.params, composed)

	if prvdr.params.Stream() {
		prvdr.handleStreamingRequest(composed)
	} else {
		prvdr.handleSingleRequest(composed)
	}

	return n, nil
}

/*
Close cleans up any resources.
*/
func (prvdr *OllamaProvider) Close() error {
	errnie.Debug("provider.OllamaProvider.Close")
	prvdr.cancel()
	return prvdr.params.Close()
}

func (prvdr *OllamaProvider) buildMessages(
	params *aiCtx.Artifact,
	chatParams *api.ChatRequest,
) {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "ollama")
		return
	}

	messages, err := params.Messages()
	if err != nil {
		errnie.Error("failed to get messages", "error", err)
		return
	}

	messageList := make([]api.Message, 0, messages.Len())

	for idx := range messages.Len() {
		message := messages.At(idx)

		role, err := message.Role()
		if err != nil {
			errnie.Error("failed to get message role", "error", err)
			continue
		}

		content, err := message.Content()
		if err != nil {
			errnie.Error("failed to get message content", "error", err)
			continue
		}

		switch role {
		case "system":
			messageList = append(messageList, api.Message{
				Role:    "system",
				Content: content,
			})
		case "user":
			messageList = append(messageList, api.Message{
				Role:    "user",
				Content: content,
			})
		case "assistant":
			messageList = append(messageList, api.Message{
				Role:    "assistant",
				Content: content,
			})
		default:
			errnie.Error("unknown message role", "role", role)
		}
	}

	chatParams.Messages = messageList
}

func (prvdr *OllamaProvider) buildTools(
	params *aiCtx.Artifact,
	chatParams *api.ChatRequest,
) {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "ollama")
		return
	}

	tools, err := params.Tools()
	if err != nil {
		errnie.Error("failed to get tools", "error", err)
		return
	}

	if tools.Len() == 0 {
		return
	}

	toolList := make([]api.Tool, 0, tools.Len())

	for idx := range tools.Len() {
		tool := tools.At(idx)

		name, err := tool.Name()
		if err != nil {
			errnie.Error("failed to get tool name", "error", err)
			continue
		}

		description, err := tool.Description()
		if err != nil {
			errnie.Error("failed to get tool description", "error", err)
			continue
		}

		toolList = append(toolList, api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        name,
				Description: description,
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
					Required: []string{"input"},
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
}

func (prvdr *OllamaProvider) buildResponseFormat(
	params *aiCtx.Artifact,
	chatParams *api.ChatRequest,
) {
	errnie.Debug("provider.buildResponseFormat")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "ollama")
		return
	}

	data, err := params.Process()
	if err != nil || data == nil {
		return
	}

	var formatData map[string]interface{}
	if err := json.Unmarshal(data, &formatData); err != nil {
		errnie.Error("failed to unmarshal process data", "error", err)
		return
	}

	// Add format instructions as a system message since Ollama doesn't support direct format control
	if name, ok := formatData["name"].(string); ok {
		if desc, ok := formatData["description"].(string); ok {
			formatMsg := api.Message{
				Role: "system",
				Content: "Please format your response according to the specified schema: " +
					name + ". " + desc,
			}
			chatParams.Messages = append(chatParams.Messages, formatMsg)
		}
	}
}

func (prvdr *OllamaProvider) handleSingleRequest(
	params *api.ChatRequest,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	err = prvdr.client.Chat(prvdr.ctx, params, func(response api.ChatResponse) error {
		return utils.SendEvent(
			prvdr.buffer,
			"provider.ollama",
			message.AssistantRole,
			response.Message.Content,
		)
	})

	return errnie.Error(err)
}

func (prvdr *OllamaProvider) handleStreamingRequest(
	params *api.ChatRequest,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream := true
	params.Stream = &stream

	return prvdr.client.Chat(prvdr.ctx, params, func(response api.ChatResponse) error {
		if response.Message.Content != "" {
			if err = utils.SendEvent(
				prvdr.buffer,
				"provider.ollama",
				message.AssistantRole,
				response.Message.Content,
			); errnie.Error(err) != nil {
				return err
			}
		}
		return nil
	})
}

type OllamaEmbedder struct {
	client *api.Client
	model  string
	ctx    context.Context
	params *aiCtx.Artifact
}

func NewOllamaEmbedder(host string) (*OllamaEmbedder, error) {
	errnie.Debug("provider.NewOllamaEmbedder")

	if host == "" {
		host = viper.GetViper().GetString("endpoints.ollama")
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		errnie.Error("failed to create Ollama embedder client", "error", err)
		return nil, err
	}

	return &OllamaEmbedder{
		client: client,
		model:  "llama2",
		ctx:    context.Background(),
		params: &aiCtx.Artifact{},
	}, nil
}

func (embedder *OllamaEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OllamaEmbedder.Read", "p", string(p))
	return 0, nil
}

func (embedder *OllamaEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OllamaEmbedder.Write")

	messages, err := embedder.params.Messages()
	if err != nil {
		errnie.Error("failed to get messages", "error", err)
		return 0, err
	}

	if messages.Len() == 0 {
		return len(p), nil
	}

	message := messages.At(0)
	content, err := message.Content()
	if err != nil {
		errnie.Error("failed to get message content", "error", err)
		return 0, err
	}

	response, err := embedder.client.Embeddings(context.Background(), &api.EmbeddingRequest{
		Model:  "llama2",
		Prompt: content,
	})
	if err != nil {
		errnie.Error("embedding request failed", "error", err)
		return 0, err
	}

	if response != nil && len(response.Embedding) > 0 {
		errnie.Debug("created embeddings",
			"text_length", len(content),
			"dimensions", len(response.Embedding),
		)
	}

	return len(p), nil
}

func (embedder *OllamaEmbedder) Close() error {
	errnie.Debug("provider.OllamaEmbedder.Close")
	embedder.params = nil
	return nil
}

func (embedder *OllamaEmbedder) Embed(text string) ([]float32, error) {
	response, err := embedder.client.Embed(embedder.ctx, &api.EmbedRequest{
		Model: embedder.model,
		Input: text,
	})
	if err != nil {
		return nil, err
	}

	// The Embed endpoint returns [][]float32, but we only need the first embedding
	if len(response.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return response.Embeddings[0], nil
}
