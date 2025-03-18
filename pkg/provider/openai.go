package provider

import (
	"context"
	"errors"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/spf13/viper"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/message"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	client *openai.Client
	buffer *stream.Buffer
	params *Params
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewOpenAIProvider creates a new OpenAI provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIProvider(
	apiKey string,
	endpoint string,
) *OpenAIProvider {
	errnie.Debug("provider.NewOpenAIProvider")

	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	if endpoint == "" {
		endpoint = viper.GetViper().GetString("endpoints.openai")
	}

	ctx, cancel := context.WithCancel(context.Background())
	params := &Params{}

	prvdr := &OpenAIProvider{
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
		),
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
func (prvdr *OpenAIProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIProvider.Read")
	return prvdr.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (prvdr *OpenAIProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIProvider.Write")

	if n, err = prvdr.buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	composed := &openai.ChatCompletionNewParams{
		Model:            openai.F(prvdr.params.Model),
		Temperature:      openai.F(prvdr.params.Temperature),
		TopP:             openai.F(prvdr.params.TopP),
		FrequencyPenalty: openai.F(prvdr.params.FrequencyPenalty),
		PresencePenalty:  openai.F(prvdr.params.PresencePenalty),
		MaxTokens:        openai.F(int64(prvdr.params.MaxTokens)),
	}

	if err = prvdr.buildMessages(composed); err != nil {
		return n, errnie.Error(err)
	}

	if err = prvdr.buildTools(composed); err != nil {
		return n, errnie.Error(err)
	}

	if err = prvdr.buildResponseFormat(composed); err != nil {
		return n, errnie.Error(err)
	}

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
func (prvdr *OpenAIProvider) Close() error {
	errnie.Debug("provider.OpenAIProvider.Close")
	return prvdr.buffer.Close()
}

/*
handleSingleRequest processes a single (non-streaming) completion request
*/
func (prvdr *OpenAIProvider) handleSingleRequest(
	params *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	var completion *openai.ChatCompletion

	if completion, err = prvdr.client.Chat.Completions.New(
		prvdr.ctx, *params,
	); errnie.Error(err) != nil {
		return err
	}

	return utils.SendEvent(
		prvdr.buffer,
		"provider.openai",
		message.AssistantRole,
		completion.Choices[0].Message.Content,
	)
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *OpenAIProvider) handleStreamingRequest(
	params *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream := prvdr.client.Chat.Completions.NewStreaming(prvdr.ctx, *params)
	defer stream.Close()

	acc := openai.ChatCompletionAccumulator{}

	for stream.Next() {
		chunk := stream.Current()

		if ok := acc.AddChunk(chunk); !ok {
			errnie.Error("chunk dropped", "id", acc.ID)
			continue
		}

		// When this fires, the current chunk value will not contain content data
		if content, ok := acc.JustFinishedContent(); ok {
			if err = utils.SendEvent(
				prvdr.buffer,
				"provider.openai",
				message.AssistantRole,
				content,
			); errnie.Error(err) != nil {
				continue
			}
		}

		// Handle delta content
		if chunk.Choices[0].Delta.Content != "" {
			if err = utils.SendEvent(
				prvdr.buffer,
				"provider.openai",
				message.AssistantRole,
				chunk.Choices[0].Delta.Content,
			); errnie.Error(err) != nil {
				continue
			}
		}
	}

	if err = stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		return
	}

	return nil
}

/*
buildMessages converts ContextData messages to OpenAI API format
*/
func (prvdr *OpenAIProvider) buildMessages(
	composed *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "openai")
		return
	}

	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(prvdr.params.Messages))

	for _, message := range prvdr.params.Messages {
		switch message.Role {
		case "system":
			messages = append(messages, openai.SystemMessage(message.Content))
		case "user":
			messages = append(messages, openai.UserMessage(message.Content))
		case "assistant":
			messages = append(messages, openai.AssistantMessage(message.Content))
		default:
			return errnie.Error(errors.New("unknown message role"))
		}
	}

	composed.Messages = openai.F(messages)

	return nil
}

/*
buildTools converts ContextData tools to OpenAI API format
*/
func (prvdr *OpenAIProvider) buildTools(
	openaiParams *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.buildTools")

	if openaiParams == nil {
		return errnie.NewErrValidation("params are nil", "provider", "openai")
	}

	toolsOut := make([]openai.ChatCompletionToolParam, 0)

	for _, tool := range prvdr.params.Tools {
		properties := make(map[string]any)

		for _, property := range tool.Function.Parameters.Properties {
			properties[property.Name] = map[string]any{
				"type":        property.Type,
				"description": property.Description,
				"enum":        property.Enum,
			}
		}

		parameters := openai.FunctionParameters{
			"type":       "object",
			"properties": properties,
			"required":   tool.Function.Parameters.Required,
		}

		toolParam := openai.ChatCompletionToolParam{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(openai.FunctionDefinitionParam{
				Name:        openai.String(tool.Function.Name),
				Description: openai.String(tool.Function.Description),
				Parameters:  openai.F(parameters),
			}),
		}

		toolsOut = append(toolsOut, toolParam)
	}

	openaiParams.Tools = openai.F(toolsOut)

	return nil
}

/*
buildResponseFormat converts ContextData response format to OpenAI API format
*/
func (prvdr *OpenAIProvider) buildResponseFormat(
	openaiParams *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	if openaiParams == nil {
		return errnie.NewErrValidation("params are nil", "provider", "openai")
	}

	openaiParams.ResponseFormat = openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](
		openai.ResponseFormatJSONSchemaParam{
			Type: openai.F(openai.ResponseFormatJSONSchemaTypeJSONSchema),
			JSONSchema: openai.F(openai.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        openai.F(prvdr.params.ResponseFormat.Name),
				Description: openai.F(prvdr.params.ResponseFormat.Description),
				Schema:      openai.F(prvdr.params.ResponseFormat.Schema),
				Strict:      openai.Bool(prvdr.params.ResponseFormat.Strict),
			}),
		},
	)

	return nil
}

/*
OpenAIEmbedder implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIEmbedder struct {
	params   *aiCtx.Artifact
	apiKey   string
	endpoint string
	client   *openai.Client
}

/*
NewOpenAIEmbedder creates a new OpenAI embedder with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIEmbedder(apiKey string, endpoint string) *OpenAIEmbedder {
	errnie.Debug("provider.NewOpenAIEmbedder")

	return &OpenAIEmbedder{
		params:   &aiCtx.Artifact{},
		apiKey:   apiKey,
		endpoint: endpoint,
		client:   openai.NewClient(option.WithAPIKey(apiKey)),
	}
}

/*
Read implements the io.Reader interface.
*/
func (embedder *OpenAIEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIEmbedder.Read", "p", string(p))
	return 0, nil
}

/*
Write implements the io.Writer interface.
*/
func (embedder *OpenAIEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIEmbedder.Write", "p", string(p))
	return len(p), nil
}

/*
Close cleans up any resources.
*/
func (embedder *OpenAIEmbedder) Close() error {
	errnie.Debug("provider.OpenAIEmbedder.Close")
	embedder.params = nil
	return nil
}
