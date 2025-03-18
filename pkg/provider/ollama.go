package provider

import (
	"context"
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/spf13/viper"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
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
	params *Params
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
	params := &Params{
		Model:       model,
		Temperature: 0.7,
		TopP:        1.0,
		MaxTokens:   2048,
	}

	client, err := api.ClientFromEnvironment()
	if errnie.Error(err) != nil {
		return nil
	}

	prvdr := &OllamaProvider{
		client: client,
		model:  model,
		buffer: stream.NewBuffer(func(artfct *datura.Artifact) (err error) {
			var payload []byte

			if payload, err = artfct.EncryptedPayload(); err != nil {
				return errnie.Error(err)
			}

			params.Unmarshal(payload)
			return nil
		}),
		params: params,
		ctx:    ctx,
		cancel: cancel,
	}

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

	prvdr.buildMessages(composed)
	prvdr.buildTools(composed)
	prvdr.buildResponseFormat(composed)

	if prvdr.params.Stream {
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
	return nil
}

func (prvdr *OllamaProvider) buildMessages(
	chatParams *api.ChatRequest,
) {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "ollama")
		return
	}

	messageList := make([]api.Message, 0, len(prvdr.params.Messages))

	for _, message := range prvdr.params.Messages {
		switch message.Role {
		case "system":
			messageList = append(messageList, api.Message{
				Role:    "system",
				Content: message.Content,
			})
		case "user":
			messageList = append(messageList, api.Message{
				Role:    "user",
				Content: message.Content,
			})
		case "assistant":
			messageList = append(messageList, api.Message{
				Role:    "assistant",
				Content: message.Content,
			})
		default:
			errnie.Error("unknown message role", "role", message.Role)
		}
	}

	chatParams.Messages = messageList
}

func (prvdr *OllamaProvider) buildTools(
	chatParams *api.ChatRequest,
) {
	errnie.Debug("provider.buildTools")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "ollama")
		return
	}

	if len(prvdr.params.Tools) == 0 {
		return
	}

	toolList := make([]api.Tool, 0, len(prvdr.params.Tools))

	for _, tool := range prvdr.params.Tools {
		toolList = append(toolList, api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
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
					Required: tool.Function.Parameters.Required,
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
	chatParams *api.ChatRequest,
) {
	errnie.Debug("provider.buildResponseFormat")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "ollama")
		return
	}

	// Add format instructions as a system message since Ollama doesn't support direct format control
	if prvdr.params.ResponseFormat.Name != "" {
		formatMsg := api.Message{
			Role: "system",
			Content: "Please format your response according to the specified schema: " +
				prvdr.params.ResponseFormat.Name + ". " + prvdr.params.ResponseFormat.Description,
		}
		chatParams.Messages = append(chatParams.Messages, formatMsg)
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
