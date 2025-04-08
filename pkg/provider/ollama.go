package provider

import (
	"context"
	"errors"
	"fmt"

	"capnproto.org/go/capnp/v3"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/ollama/ollama/api"
	"github.com/spf13/viper"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/message"
	"github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
OllamaProvider implements an LLM provider that connects to Ollama's API.
It supports regular chat completions and streaming responses.
*/
type OllamaProvider struct {
	client   *api.Client
	endpoint string
	pctx     context.Context
	ctx      context.Context
	cancel   context.CancelFunc
	segment  *capnp.Segment
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
		pctx:     ctx,
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
	p params.Params,
	ctx aicontext.Context,
	tools []mcp.Tool,
) chan *datura.Artifact {
	model, err := p.Model()

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		composed := &api.ChatRequest{
			Model: model,
			Options: map[string]interface{}{
				"temperature": p.Temperature(),
				"top_p":       p.TopP(),
				"max_tokens":  p.MaxTokens(),
			},
		}

		if err = prvdr.buildMessages(composed, ctx); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildTools(composed, tools); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		format, err := p.Format()

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildResponseFormat(composed, format); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if p.Stream() {
			prvdr.handleStreamingRequest(composed, out)
		} else {
			prvdr.handleSingleRequest(composed, out)
		}
	}()

	return out
}

func (prvdr *OllamaProvider) Name() string {
	return "ollama"
}

func (prvdr *OllamaProvider) handleSingleRequest(
	params *api.ChatRequest,
	channel chan *datura.Artifact,
) {
	errnie.Debug("provider.handleSingleRequest")

	err := prvdr.client.Chat(
		prvdr.ctx, params, func(response api.ChatResponse) error {
			// Create a new message using Cap'n Proto
			msg, err := message.NewMessage(prvdr.segment)
			if errnie.Error(err) != nil {
				return err
			}

			// Set message fields
			if err = msg.SetRole("assistant"); errnie.Error(err) != nil {
				return err
			}

			if err = msg.SetName(params.Model); errnie.Error(err) != nil {
				return err
			}

			if err = msg.SetContent(response.Message.Content); errnie.Error(err) != nil {
				return err
			}

			// Create artifact with message content
			channel <- datura.New(datura.WithEncryptedPayload([]byte(response.Message.Content)))
			return nil
		},
	)

	if errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
	}
}

func (prvdr *OllamaProvider) handleStreamingRequest(
	params *api.ChatRequest,
	channel chan *datura.Artifact,
) {
	errnie.Debug("provider.handleStreamingRequest")

	stream := true
	params.Stream = &stream

	err := prvdr.client.Chat(prvdr.ctx, params, func(response api.ChatResponse) error {
		if response.Message.Content != "" {
			channel <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithEncryptedPayload([]byte(response.Message.Content)),
			)
		}
		return nil
	})

	if errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
	}
}

func (prvdr *OllamaProvider) buildMessages(
	chatParams *api.ChatRequest,
	ctx aicontext.Context,
) (err error) {
	errnie.Debug("provider.buildMessages")

	msgs, err := ctx.Messages()

	if errnie.Error(err) != nil {
		return err
	}

	messageList := make([]api.Message, 0, msgs.Len())

	for i := 0; i < msgs.Len(); i++ {
		msg := msgs.At(i)

		role, err := msg.Role()
		if errnie.Error(err) != nil {
			return err
		}

		content, err := msg.Content()
		if errnie.Error(err) != nil {
			return err
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
			// Check for tool calls but don't store the variable since we don't use it
			if _, err := msg.ToolCalls(); errnie.Error(err) != nil {
				return err
			}

			// For Ollama, we currently don't handle tool calls in history
			// But we can still add the assistant message
			messageList = append(messageList, api.Message{
				Role:    "assistant",
				Content: content,
			})
		case "tool":
			// Ollama doesn't have a tool message type, so we'll add it as a user message
			// with the content indicating it's a tool result
			id, err := msg.Id()
			if errnie.Error(err) != nil {
				return err
			}

			messageList = append(messageList, api.Message{
				Role:    "user",
				Content: fmt.Sprintf("[Tool Result from %s: %s]", id, content),
			})
		default:
			errnie.Error("unknown message role", "role", role)
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
	format params.ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	// Extract format details
	name, err := format.Name()
	if errnie.Error(err) != nil {
		return err
	}

	description, err := format.Description()
	if errnie.Error(err) != nil {
		return err
	}

	schema, err := format.Schema()
	if errnie.Error(err) != nil {
		return err
	}

	// Add format instructions as a system message since Ollama doesn't support direct format control
	if name != "" || description != "" || schema != "" {
		formatMsg := api.Message{
			Role: "system",
			Content: "Please format your response according to the specified schema: " +
				name + ". " + description + "\nSchema: " + schema,
		}
		chatParams.Messages = append(chatParams.Messages, formatMsg)
	}

	return nil
}

type OllamaEmbedder struct {
	client   *api.Client
	endpoint string
	ctx      context.Context
	cancel   context.CancelFunc
}

func NewOllamaEmbedder(opts ...OllamaEmbedderOption) *OllamaEmbedder {
	errnie.Debug("provider.NewOllamaEmbedder")

	endpoint := viper.GetViper().GetString("endpoints.ollama")
	ctx, cancel := context.WithCancel(context.Background())

	client, err := api.ClientFromEnvironment()
	if err != nil {
		errnie.Error("failed to create Ollama embedder client", "error", err)
		cancel()
		return nil
	}

	embedder := &OllamaEmbedder{
		client:   client,
		endpoint: endpoint,
		ctx:      ctx,
		cancel:   cancel,
	}

	for _, opt := range opts {
		opt(embedder)
	}

	return embedder
}

type OllamaEmbedderOption func(*OllamaEmbedder)

func WithOllamaEmbedderEndpoint(endpoint string) OllamaEmbedderOption {
	return func(embedder *OllamaEmbedder) {
		embedder.endpoint = endpoint
	}
}

func (embedder *OllamaEmbedder) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("provider.OllamaEmbedder.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-embedder.ctx.Done():
			errnie.Debug("provider.OllamaEmbedder.Generate.ctx.Done")
			embedder.cancel()
			return
		case artifact := <-buffer:
			content, err := artifact.DecryptPayload()
			if err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			if len(content) == 0 {
				out <- datura.New(datura.WithError(errnie.Error(errors.New("content is empty"))))
				return
			}

			embeddings, err := embedder.Embed(string(content))
			if err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			// Convert embeddings to bytes (this is a simplified version)
			embeddingsBytes := make([]byte, len(embeddings)*4)
			out <- datura.New(datura.WithEncryptedPayload(embeddingsBytes))
		}
	}()

	return out
}

func (embedder *OllamaEmbedder) Close() error {
	errnie.Debug("provider.OllamaEmbedder.Close")
	embedder.cancel()
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
