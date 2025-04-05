package provider

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"

	"capnproto.org/go/capnp/v3"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	client  *openai.Client
	pctx    context.Context
	ctx     context.Context
	cancel  context.CancelFunc
	segment *capnp.Segment
}

/*
NewOpenAIProvider creates a new OpenAI provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIProvider(opts ...OpenAIProviderOption) *OpenAIProvider {
	errnie.Debug("provider.NewOpenAIProvider")

	apiKey := os.Getenv("OPENAI_API_KEY")

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	// Initialize a new segment
	arena := capnp.SingleSegment(nil)
	_, segment, err := capnp.NewMessage(arena)
	if err != nil {
		errnie.Error(err)
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &OpenAIProvider{
		client:  &client,
		pctx:    ctx,
		ctx:     ctx,
		cancel:  cancel,
		segment: segment,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *OpenAIProvider) ID() string {
	return "openai"
}

func (prvdr *OpenAIProvider) Generate(
	artifact *datura.ArtifactBuilder,
) *datura.ArtifactBuilder {
	composed := &openai.ChatCompletionNewParams{
		Model:            openai.ChatModel(datura.GetMetaValue[string](artifact, "model")),
		Temperature:      openai.Float(datura.GetMetaValue[float64](artifact, "temperature")),
		TopP:             openai.Float(datura.GetMetaValue[float64](artifact, "top_p")),
		FrequencyPenalty: openai.Float(datura.GetMetaValue[float64](artifact, "frequency_penalty")),
		PresencePenalty:  openai.Float(datura.GetMetaValue[float64](artifact, "presence_penalty")),
	}

	if datura.GetMetaValue[int](artifact, "max_tokens") > 1 {
		composed.MaxTokens = openai.Int(int64(
			datura.GetMetaValue[int](artifact, "max_tokens"),
		))
	}

	var err error

	if err = prvdr.buildMessages(composed, artifact); err != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	// Get tools from the artifact metadata
	toolsData := datura.GetMetaValue[[]tools.ToolType](artifact, "tools")
	if err = prvdr.buildTools(composed, toolsData); err != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	format := datura.GetMetaValue[string](artifact, "format")

	if format != "" {
		if err = prvdr.buildResponseFormat(composed, format); err != nil {
			return datura.New(datura.WithError(errnie.Error(err)))
		}
	}

	if datura.GetMetaValue[bool](artifact, "stream") {
		return prvdr.handleStreamingRequest(composed)
	}

	return prvdr.handleSingleRequest(composed)
}

func (prvdr *OpenAIProvider) Name() string {
	return "openai"
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
handleSingleRequest processes a single (non-streaming) completion request
*/
func (prvdr *OpenAIProvider) handleSingleRequest(
	params *openai.ChatCompletionNewParams,
) *datura.ArtifactBuilder {
	errnie.Debug("provider.handleSingleRequest")

	var (
		err        error
		completion *openai.ChatCompletion
	)

	if completion, err = prvdr.client.Chat.Completions.New(
		prvdr.ctx, *params,
	); errnie.Error(err) != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	// Create a new message using the provider's segment
	msg, err := aicontext.NewMessage(prvdr.segment)
	if errnie.Error(err) != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	// Set message fields
	if err = msg.SetRole("assistant"); errnie.Error(err) != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	if err = msg.SetName(params.Model); errnie.Error(err) != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	if err = msg.SetContent(completion.Choices[0].Message.Content); errnie.Error(err) != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	toolCalls := completion.Choices[0].Message.ToolCalls

	// Abort early if there are no tool calls
	if len(toolCalls) == 0 {
		return datura.New(datura.WithPayload([]byte(completion.Choices[0].Message.Content)))
	}

	// Create tool calls list
	toolCallList, err := msg.NewToolCalls(int32(len(toolCalls)))
	if errnie.Error(err) != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	for i, toolCall := range toolCalls {
		errnie.Info("toolCall", "tool", toolCall.Function.Name, "id", toolCall.ID)

		if err = toolCallList.At(i).SetId(toolCall.ID); errnie.Error(err) != nil {
			return datura.New(datura.WithError(errnie.Error(err)))
		}

		if err = toolCallList.At(i).SetName(toolCall.Function.Name); errnie.Error(err) != nil {
			return datura.New(datura.WithError(errnie.Error(err)))
		}

		if err = toolCallList.At(i).SetArguments(toolCall.Function.Arguments); errnie.Error(err) != nil {
			return datura.New(datura.WithError(errnie.Error(err)))
		}
	}

	// Create artifact with message content
	return datura.New(datura.WithPayload([]byte(completion.Choices[0].Message.Content)))
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *OpenAIProvider) handleStreamingRequest(
	params *openai.ChatCompletionNewParams,
) *datura.ArtifactBuilder {
	errnie.Debug("provider.handleStreamingRequest")

	var err error
	stream := prvdr.client.Chat.Completions.NewStreaming(prvdr.ctx, *params)
	defer stream.Close()

	acc := openai.ChatCompletionAccumulator{}

	for stream.Next() {
		chunk := stream.Current()

		if ok := acc.AddChunk(chunk); !ok {
			return datura.New(datura.WithError(err))
		}

		if content, ok := acc.JustFinishedContent(); ok && content != "" {
			return datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(content)),
			)
		}

		if tool, ok := acc.JustFinishedToolCall(); ok {
			params.Messages = append(params.Messages, openai.AssistantMessage(acc.Choices[0].Message.Content))
			return datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(tool.Arguments)),
			)
		}

		if refusal, ok := acc.JustFinishedRefusal(); ok && refusal != "" {
			return datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(refusal)),
			)
		}

		// Only write non-empty content from chunks
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			return datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(chunk.Choices[0].Delta.Content)),
			)
		}
	}

	if err = stream.Err(); err != nil {
		return datura.New(
			datura.WithError(errnie.Error("Streaming error", "error", err)),
		)
	}

	return nil
}

/*
buildMessages converts ContextData messages to OpenAI API format
*/
func (prvdr *OpenAIProvider) buildMessages(
	composed *openai.ChatCompletionNewParams,
	artifact *datura.ArtifactBuilder,
) (err error) {
	errnie.Debug("provider.buildMessages")

	payload, err := artifact.DecryptPayload()
	if errnie.Error(err) != nil {
		return err
	}

	messages := []OpenAIMessage{}
	if err := json.Unmarshal(payload, &messages); errnie.Error(err) != nil {
		return err
	}

	openaiMessages := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			openaiMessages = append(openaiMessages, openai.SystemMessage(msg.Content))
		case "user":
			openaiMessages = append(openaiMessages, openai.UserMessage(msg.Content))
		case "assistant":
			tcs := make([]openai.ChatCompletionMessageToolCallParam, 0, len(msg.ToolCalls))

			for _, toolCall := range msg.ToolCalls {
				id, err := toolCall.Id()
				if errnie.Error(err) != nil {
					return err
				}

				name, err := toolCall.Name()
				if errnie.Error(err) != nil {
					return err
				}

				arguments, err := toolCall.Arguments()
				if errnie.Error(err) != nil {
					return err
				}

				tcs = append(tcs, openai.ChatCompletionMessageToolCallParam{
					ID:   id,
					Type: "function",
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      name,
						Arguments: arguments,
					},
				})
			}

			assistantMsg := openai.AssistantMessage(msg.Content)
			if len(tcs) > 0 {
				assistantMsg = openai.ChatCompletionMessageParamUnion{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfString: param.NewOpt(msg.Content),
						},
						ToolCalls: tcs,
						Role:      "assistant",
					},
				}
			}

			openaiMessages = append(openaiMessages, assistantMsg)
		case "tool":
			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: param.NewOpt(msg.Content),
					},
					ToolCallID: msg.ToolCallID,
					Role:       "tool",
				},
			})
		default:
			return errnie.Error(errors.New("unknown message role"), "role", msg.Role)
		}
	}

	composed.Messages = openaiMessages
	return nil
}

/*
buildTools takes the tools from the generic params and converts them to OpenAI API format.
It is important to return nil early when there are no tools, because passing an empty array
to the OpenAI API will cause strange behavior, like the model guessing random tools.
*/
func (prvdr *OpenAIProvider) buildTools(
	openaiParams *openai.ChatCompletionNewParams,
	tools []tools.ToolType,
) (err error) {
	errnie.Debug("provider.buildTools")

	if openaiParams == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	if len(tools) == 0 {
		// No tools, no shoes, no dice.
		return nil
	}

	toolsOut := make([]openai.ChatCompletionToolParam, 0, len(tools))

	for _, tool := range tools {
		toolParam := openai.ChatCompletionToolParam{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Tool.Name,
				Description: param.NewOpt(tool.Tool.Description),
				Parameters: openai.FunctionParameters{
					"type":       tool.Tool.InputSchema.Type,
					"properties": tool.Tool.InputSchema.Properties,
				},
			},
		}
		toolsOut = append(toolsOut, toolParam)
	}

	if len(toolsOut) > 0 {
		openaiParams.Tools = toolsOut
	}

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
	format string,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	buf := map[string]any{}

	if err = json.Unmarshal([]byte(format), &buf); errnie.Error(err) != nil {
		return err
	}

	openaiParams.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			Type: "json_schema",
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        buf["name"].(string),
				Description: param.NewOpt(buf["description"].(string)),
				Schema:      buf["schema"].(map[string]any),
				Strict:      param.NewOpt(buf["strict"].(bool)),
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
	client *openai.Client
	ctx    context.Context
	cancel context.CancelFunc
}

type OpenAIEmbedderOption func(*OpenAIEmbedder)

/*
NewOpenAIEmbedder creates a new OpenAI embedder with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIEmbedder(opts ...OpenAIEmbedderOption) *OpenAIEmbedder {
	errnie.Debug("provider.NewOpenAIEmbedder")

	apiKey := os.Getenv("OPENAI_API_KEY")
	ctx, cancel := context.WithCancel(context.Background())

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	embedder := &OpenAIEmbedder{
		client: &client,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(embedder)
	}

	return embedder
}

/*
Generate implements the Generator interface for OpenAIEmbedder.
It takes input text through a channel and returns embeddings through another channel.
*/
func (embedder *OpenAIEmbedder) Generate(
	artifact *datura.ArtifactBuilder,
) *datura.ArtifactBuilder {
	errnie.Debug("provider.OpenAIEmbedder.Generate")

	content, err := artifact.DecryptPayload()
	if err != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	if len(content) == 0 {
		return datura.New(datura.WithError(errnie.Error(errors.New("content is empty"))))
	}

	response, err := embedder.client.Embeddings.New(embedder.ctx, openai.EmbeddingNewParams{
		Input:          openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: []string{string(content)}},
		Model:          openai.EmbeddingModelTextEmbeddingAda002,
		Dimensions:     openai.Int(tweaker.GetQdrantDimension()),
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	})
	if err != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	if len(response.Data) == 0 {
		return datura.New(datura.WithError(errnie.Error(errors.New("no embeddings returned"))))
	}

	// Convert float64 embeddings to float32
	embeddings := response.Data[0].Embedding
	float32Embeddings := make([]float32, len(embeddings))
	for i, v := range embeddings {
		float32Embeddings[i] = float32(v)
	}

	// Convert embeddings to bytes
	embeddingsBytes := make([]byte, len(float32Embeddings)*4)
	for i, v := range float32Embeddings {
		binary.LittleEndian.PutUint32(embeddingsBytes[i*4:], math.Float32bits(v))
	}

	return datura.New(datura.WithPayload(embeddingsBytes))
}

func WithOpenAIEmbedderAPIKey(apiKey string) OpenAIEmbedderOption {
	return func(embedder *OpenAIEmbedder) {
		embedder.client.Options = append(embedder.client.Options, option.WithAPIKey(apiKey))
	}
}

func WithOpenAIEmbedderEndpoint(endpoint string) OpenAIEmbedderOption {
	return func(embedder *OpenAIEmbedder) {
		embedder.client.Options = append(embedder.client.Options, option.WithBaseURL(endpoint))
	}
}
