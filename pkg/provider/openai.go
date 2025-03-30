package provider

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tweaker"
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
func NewOpenAIProvider(opts ...OpenAIProviderOption) *OpenAIProvider {
	errnie.Debug("provider.NewOpenAIProvider")

	apiKey := os.Getenv("OPENAI_API_KEY")

	ctx, cancel := context.WithCancel(context.Background())
	params := &Params{}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	prvdr := &OpenAIProvider{
		client: &client,
		buffer: stream.NewBuffer(func(artfct *datura.Artifact) (err error) {
			errnie.Debug("provider.OpenAIProvider.buffer.fn")
			return errnie.Error(artfct.To(params))
		}),
		params: params,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

type OpenAIProviderOption func(*OpenAIProvider)

func WithAPIKey(apiKey string) OpenAIProviderOption {
	return func(prvdr *OpenAIProvider) {
		prvdr.client.Options = append(prvdr.client.Options, option.WithAPIKey(apiKey))
	}
}

func WithEndpoint(endpoint string) OpenAIProviderOption {
	return func(prvdr *OpenAIProvider) {
		prvdr.client.Options = append(prvdr.client.Options, option.WithBaseURL(endpoint))
	}
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
		Model:            prvdr.params.Model,
		Temperature:      openai.Float(prvdr.params.Temperature),
		TopP:             openai.Float(prvdr.params.TopP),
		FrequencyPenalty: openai.Float(prvdr.params.FrequencyPenalty),
		PresencePenalty:  openai.Float(prvdr.params.PresencePenalty),
	}

	if prvdr.params.MaxTokens > 1 {
		composed.MaxTokens = openai.Int(int64(prvdr.params.MaxTokens))
	}

	if err = prvdr.buildMessages(composed); err != nil {
		return n, errnie.Error(err)
	}

	if err = prvdr.buildTools(composed); err != nil {
		return n, errnie.Error(err)
	}

	if prvdr.params.ResponseFormat != nil {
		if err = prvdr.buildResponseFormat(composed); err != nil {
			return n, errnie.Error(err)
		}
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

	msg := &Message{
		Role:    MessageRoleAssistant,
		Name:    prvdr.params.Model,
		Content: completion.Choices[0].Message.Content,
	}

	toolCalls := completion.Choices[0].Message.ToolCalls

	// Abort early if there are no tool calls
	if len(toolCalls) == 0 {
		prvdr.params.Messages = append(prvdr.params.Messages, msg)

		if _, err = io.Copy(prvdr.buffer, datura.New(
			datura.WithPayload(prvdr.params.Marshal()),
		)); err != nil {
			return errnie.Error(err)
		}

		return nil
	}

	msg.ToolCalls = make([]ToolCall, 0, len(toolCalls))

	for _, toolCall := range toolCalls {
		errnie.Info("toolCall", "tool", toolCall.Function.Name, "id", toolCall.ID)

		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:   toolCall.ID,
			Type: "function",
			Function: ToolCallFunction{
				Name:      toolCall.Function.Name,
				Arguments: toolCall.Function.Arguments,
			},
		})
	}

	prvdr.params.Messages = append(prvdr.params.Messages, msg)

	if _, err = io.Copy(prvdr.buffer, datura.New(
		datura.WithPayload(prvdr.params.Marshal()),
	)); err != nil {
		return errnie.Error(err)
	}

	return nil
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

		if content, ok := acc.JustFinishedContent(); ok && content != "" {
			if _, err = io.Copy(prvdr.buffer, datura.New(
				datura.WithPayload([]byte(content)),
			)); errnie.Error(err) != nil {
				continue
			}
		}

		if tool, ok := acc.JustFinishedToolCall(); ok {
			params.Messages = append(params.Messages, openai.AssistantMessage(acc.Choices[0].Message.Content))

			switch tool.Name {
			case "browser":
				if tool.Arguments != "" {
					if _, err = io.Copy(prvdr.buffer, datura.New(
						datura.WithPayload([]byte(tool.Arguments)),
					)); errnie.Error(err) != nil {
						continue
					}
				}
			}
		}

		if refusal, ok := acc.JustFinishedRefusal(); ok && refusal != "" {
			if _, err = io.Copy(prvdr.buffer, datura.New(
				datura.WithPayload([]byte(refusal)),
			)); errnie.Error(err) != nil {
				continue
			}
		}

		// Only write non-empty content from chunks
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			if _, err = io.Copy(prvdr.buffer, datura.New(
				datura.WithPayload([]byte(chunk.Choices[0].Delta.Content)),
			)); errnie.Error(err) != nil {
				continue
			}
		}
	}

	if err = stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		return err
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
		return errnie.BadRequest(errors.New("params are nil"))
	}

	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(prvdr.params.Messages))

	for _, message := range prvdr.params.Messages {
		switch message.Role {
		case "system":
			messages = append(messages, openai.SystemMessage(message.Content))
		case "user":
			messages = append(messages, openai.UserMessage(message.Content))
		case "assistant":
			toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0, len(message.ToolCalls))

			for _, toolCall := range message.ToolCalls {
				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
					ID:   toolCall.ID,
					Type: "function",
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				})
			}

			errnie.Info("toolCalls", "toolCalls", toolCalls)

			msg := openai.AssistantMessage(message.Content)
			if len(toolCalls) > 0 {
				msg = openai.ChatCompletionMessageParamUnion{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfString: param.NewOpt(message.Content),
						},
						ToolCalls: toolCalls,
						Role:      "assistant",
					},
				}
			}

			messages = append(messages, msg)
		case "tool":
			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: param.NewOpt(message.Content),
					},
					ToolCallID: message.Reference,
					Role:       "tool",
				},
			})
		default:
			fmt.Println(message.Role, message.Content)
			return errnie.Error(errors.New("unknown message role"))
		}
	}

	composed.Messages = messages

	return nil
}

/*
buildTools takes the tools from the generic params and converts them to OpenAI API format.
It is important to return nil early when there are no tools, because passing an empty array
to the OpenAI API will cause strange behavior, like the model guessing random tools.
*/
func (prvdr *OpenAIProvider) buildTools(
	openaiParams *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.buildTools")

	if openaiParams == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	if len(prvdr.params.Tools) == 0 {
		// No tools, no shoes, no dice.
		return nil
	}

	toolsOut := make([]openai.ChatCompletionToolParam, 0)

	for _, tool := range prvdr.params.Tools {
		properties := make(map[string]any)

		for _, property := range tool.Function.Parameters.Properties {
			propDef := map[string]any{
				"type":        property.Type,
				"description": property.Description,
			}
			if len(property.Enum) > 0 {
				propDef["enum"] = property.Enum
			}
			properties[property.Name] = propDef
		}

		// Ensure we always have a valid parameters object that matches OpenAI's schema
		parameters := openai.FunctionParameters{
			"type":       "object",
			"properties": properties,
		}

		// Only include required if it has values
		if len(tool.Function.Parameters.Required) > 0 {
			parameters["required"] = tool.Function.Parameters.Required
		} else {
			parameters["required"] = []string{}
		}

		toolParam := openai.ChatCompletionToolParam{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Function.Name,
				Description: param.NewOpt(tool.Function.Description),
				Parameters:  parameters,
			},
		}

		toolsOut = append(toolsOut, toolParam)
	}

	openaiParams.Tools = toolsOut

	return nil
}

/*
buildResponseFormat converts the response format from the generic params to OpenAI API format.
This will force the model to use structured output, and return a JSON object.
Setting Strict to true will make sure the only thing returned is the JSON object.
If you want this to be combined with the ability to call tools, you can set Strict to false.
*/
func (prvdr *OpenAIProvider) buildResponseFormat(
	openaiParams *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	if openaiParams == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	openaiParams.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			Type: "json_schema",
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        prvdr.params.ResponseFormat.Name,
				Description: param.NewOpt(prvdr.params.ResponseFormat.Description),
				Schema:      prvdr.params.ResponseFormat.Schema,
				Strict:      param.NewOpt(prvdr.params.ResponseFormat.Strict),
			},
		},
	}

	return nil
}

/*
OpenAIEmbedder implements an LLM provider that connects to OpenAI's API.
It supports embedding generation for text inputs.
*/
type OpenAIEmbedder struct {
	client     *openai.Client
	buffer     *stream.Buffer
	params     *Params
	embeddings []float32
}

type OpenAIEmbedderOption func(*OpenAIEmbedder)

/*
NewOpenAIEmbedder creates a new OpenAI embedder with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIEmbedder(opts ...OpenAIEmbedderOption) *OpenAIEmbedder {
	errnie.Debug("provider.NewOpenAIEmbedder")

	params := &Params{}
	client := openai.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))

	embedder := &OpenAIEmbedder{
		params: params,
		client: &client,
	}

	embedder.buffer = stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
		errnie.Debug("provider.OpenAIEmbedder.buffer.fn")

		var content []byte

		if content, err = artifact.DecryptPayload(); err != nil {
			return errnie.Error(err)
		}

		if len(content) == 0 {
			return errnie.Error(errors.New("content is empty"))
		}

		var response *openai.CreateEmbeddingResponse

		// Get embeddings from OpenAI
		if response, err = client.Embeddings.New(context.TODO(), openai.EmbeddingNewParams{
			Input:          openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: []string{string(content)}},
			Model:          openai.EmbeddingModelTextEmbeddingAda002,
			Dimensions:     openai.Int(tweaker.GetQdrantDimension()),
			EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
		}); errnie.Error(err) != nil {
			return err
		}

		if len(response.Data) == 0 {
			return errnie.Error(errors.New("no embeddings returned"))
		}

		// Convert float64 embeddings to float32
		embeddings := response.Data[0].Embedding
		embedder.embeddings = make([]float32, len(embeddings))
		for i, v := range embeddings {
			embedder.embeddings[i] = float32(v)
		}

		// Convert embeddings to bytes and store in artifact payload
		embeddingsBytes := make([]byte, len(embedder.embeddings)*4)

		for i, v := range embedder.embeddings {
			binary.LittleEndian.PutUint32(embeddingsBytes[i*4:], math.Float32bits(v))
		}

		datura.WithPayload(embeddingsBytes)(artifact)
		return nil
	})

	for _, opt := range opts {
		opt(embedder)
	}

	return embedder
}

/*
Read implements the io.Reader interface.
*/
func (embedder *OpenAIEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIEmbedder.Read")
	return embedder.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (embedder *OpenAIEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIEmbedder.Write")
	return embedder.buffer.Write(p)
}

/*
Close implements io.Closer for OpenAIEmbedder.
*/
func (embedder *OpenAIEmbedder) Close() error {
	errnie.Debug("provider.OpenAIEmbedder.Close")
	embedder.params = nil
	embedder.embeddings = nil
	return embedder.buffer.Close()
}

func WithOpenAIEmbedderAPIKey(apiKey string) OpenAIEmbedderOption {
	return func(embedder *OpenAIEmbedder) {
		client := openai.NewClient(option.WithAPIKey(apiKey))
		embedder.client = &client
	}
}

func WithOpenAIEmbedderEndpoint(endpoint string) OpenAIEmbedderOption {
	return func(embedder *OpenAIEmbedder) {
		client := openai.NewClient(option.WithBaseURL(endpoint))
		embedder.client = &client
	}
}
