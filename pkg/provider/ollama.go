package provider

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/ollama/ollama/api"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
OllamaProvider implements an LLM provider that connects to Ollama's API.
It supports regular chat completions and streaming responses.
*/
type OllamaProvider struct {
	client   *api.Client
	endpoint string
	buffer   *stream.Buffer
	params   *Params
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
		buffer:   stream.NewBuffer(),
		params:   &Params{},
		ctx:      ctx,
		cancel:   cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

type OllamaProviderOption func(*OllamaProvider)

func WithOllamaEndpoint(endpoint string) OllamaProviderOption {
	return func(prvdr *OllamaProvider) {
		prvdr.endpoint = endpoint
	}
}

func (prvdr *OllamaProvider) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	errnie.Debug("provider.OllamaProvider.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-prvdr.ctx.Done():
			errnie.Debug("provider.OllamaProvider.Generate.ctx.Done")
			prvdr.cancel()
			return
		case artifact := <-buffer:
			if err := artifact.To(prvdr.params); err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			composed := &api.ChatRequest{
				Model: prvdr.params.Model,
				Options: map[string]interface{}{
					"temperature": prvdr.params.Temperature,
					"top_p":       prvdr.params.TopP,
					"max_tokens":  prvdr.params.MaxTokens,
				},
			}

			if err := prvdr.buildMessages(composed); err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			if err := prvdr.buildTools(composed); err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			if prvdr.params.ResponseFormat != nil {
				if err := prvdr.buildResponseFormat(composed); err != nil {
					out <- datura.New(datura.WithError(err))
					return
				}
			}

			var err error
			if prvdr.params.Stream {
				err = prvdr.handleStreamingRequest(composed)
			} else {
				err = prvdr.handleSingleRequest(composed)
			}

			if err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			out <- datura.New(datura.WithPayload(prvdr.params.Marshal()))
		}
	}()

	return out
}

func (prvdr *OllamaProvider) Name() string {
	return "ollama"
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
		Model: prvdr.params.Model,
		Options: map[string]interface{}{
			"temperature": prvdr.params.Temperature,
			"top_p":       prvdr.params.TopP,
			"max_tokens":  prvdr.params.MaxTokens,
		},
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

func (prvdr *OllamaProvider) handleSingleRequest(
	params *api.ChatRequest,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	return errnie.Error(prvdr.client.Chat(
		prvdr.ctx, params, func(response api.ChatResponse) error {
			if _, err = io.Copy(prvdr, datura.New(
				datura.WithPayload([]byte(response.Message.Content)),
			)); errnie.Error(err) != nil {
				return err
			}
			return nil
		},
	))
}

func (prvdr *OllamaProvider) handleStreamingRequest(
	params *api.ChatRequest,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream := true
	params.Stream = &stream

	return prvdr.client.Chat(prvdr.ctx, params, func(response api.ChatResponse) error {
		if response.Message.Content != "" {
			if _, err = io.Copy(prvdr, datura.New(
				datura.WithPayload([]byte(response.Message.Content)),
			)); errnie.Error(err) != nil {
				return err
			}
		}
		return nil
	})
}

func (prvdr *OllamaProvider) buildMessages(
	chatParams *api.ChatRequest,
) (err error) {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
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

	return nil
}

func (prvdr *OllamaProvider) buildTools(
	chatParams *api.ChatRequest,
) (err error) {
	errnie.Debug("provider.buildTools")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
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

	return nil
}

func (prvdr *OllamaProvider) buildResponseFormat(
	chatParams *api.ChatRequest,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
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

	return nil
}

type OllamaEmbedder struct {
	client   *api.Client
	endpoint string
	ctx      context.Context
	params   *Params
}

func NewOllamaEmbedder(endpoint string) (*OllamaEmbedder, error) {
	errnie.Debug("provider.NewOllamaEmbedder")

	if endpoint == "" {
		endpoint = viper.GetViper().GetString("endpoints.ollama")
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		errnie.Error("failed to create Ollama embedder client", "error", err)
		return nil, err
	}

	return &OllamaEmbedder{
		client:   client,
		endpoint: endpoint,
		ctx:      context.Background(),
		params:   &Params{},
	}, nil
}

func (embedder *OllamaEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OllamaEmbedder.Read", "p", string(p))
	return 0, nil
}

func (embedder *OllamaEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OllamaEmbedder.Write")

	if len(embedder.params.Messages) == 0 {
		return len(p), nil
	}

	message := embedder.params.Messages[0]
	content := message.Content

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
		Model: "llama2",
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
