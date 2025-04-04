package provider

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"

	"capnproto.org/go/capnp/v3"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
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

	ctx, cancel := context.WithCancel(context.Background())

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	prvdr := &OpenAIProvider{
		client: &client,
		pctx:   ctx,
		ctx:    ctx,
		cancel: cancel,
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
	params params.Params,
	ctx aicontext.Context,
	tools []mcp.Tool,
) chan *datura.Artifact {
	model, err := params.Model()

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		composed := &openai.ChatCompletionNewParams{
			Model:            openai.ChatModel(model),
			Temperature:      openai.Float(params.Temperature()),
			TopP:             openai.Float(params.TopP()),
			FrequencyPenalty: openai.Float(params.FrequencyPenalty()),
			PresencePenalty:  openai.Float(params.PresencePenalty()),
		}

		if params.MaxTokens() > 1 {
			composed.MaxTokens = openai.Int(int64(params.MaxTokens()))
		}

		if err = prvdr.buildMessages(composed, ctx); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildTools(composed, tools); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		format, err := params.Format()

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildResponseFormat(composed, format); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if params.Stream() {
			prvdr.handleStreamingRequest(composed, out)
		} else {
			prvdr.handleSingleRequest(composed, out)
		}
	}()

	return out
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
	channel chan *datura.Artifact,
) {
	errnie.Debug("provider.handleSingleRequest")

	var (
		err        error
		completion *openai.ChatCompletion
	)

	if completion, err = prvdr.client.Chat.Completions.New(
		prvdr.ctx, *params,
	); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	// Create a new message using Cap'n Proto
	msg, err := aicontext.NewMessage(prvdr.segment)
	if errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	// Set message fields
	if err = msg.SetRole("assistant"); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	if err = msg.SetName(params.Model); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	if err = msg.SetContent(completion.Choices[0].Message.Content); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	toolCalls := completion.Choices[0].Message.ToolCalls

	// Abort early if there are no tool calls
	if len(toolCalls) == 0 {
		channel <- datura.New(datura.WithPayload([]byte(completion.Choices[0].Message.Content)))
		return
	}

	// Create tool calls list
	toolCallList, err := msg.NewToolCalls(int32(len(toolCalls)))
	if errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	for i, toolCall := range toolCalls {
		errnie.Info("toolCall", "tool", toolCall.Function.Name, "id", toolCall.ID)

		if err = toolCallList.At(i).SetId(toolCall.ID); errnie.Error(err) != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = toolCallList.At(i).SetName(toolCall.Function.Name); errnie.Error(err) != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = toolCallList.At(i).SetArguments(toolCall.Function.Arguments); errnie.Error(err) != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}
	}

	// Create artifact with message content
	channel <- datura.New(datura.WithPayload([]byte(completion.Choices[0].Message.Content)))
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *OpenAIProvider) handleStreamingRequest(
	params *openai.ChatCompletionNewParams,
	channel chan *datura.Artifact,
) {
	errnie.Debug("provider.handleStreamingRequest")

	var err error
	stream := prvdr.client.Chat.Completions.NewStreaming(prvdr.ctx, *params)
	defer stream.Close()

	acc := openai.ChatCompletionAccumulator{}

	for stream.Next() {
		chunk := stream.Current()

		if ok := acc.AddChunk(chunk); !ok {
			channel <- datura.New(datura.WithError(err))
			continue
		}

		if content, ok := acc.JustFinishedContent(); ok && content != "" {
			channel <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(content)),
			)
		}

		if tool, ok := acc.JustFinishedToolCall(); ok {
			params.Messages = append(params.Messages, openai.AssistantMessage(acc.Choices[0].Message.Content))
			channel <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(tool.Arguments)),
			)
		}

		if refusal, ok := acc.JustFinishedRefusal(); ok && refusal != "" {
			channel <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(refusal)),
			)
		}

		// Only write non-empty content from chunks
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			channel <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(chunk.Choices[0].Delta.Content)),
			)
		}
	}

	if err = stream.Err(); err != nil {
		channel <- datura.New(
			datura.WithError(errnie.Error("Streaming error", "error", err)),
		)
	}
}

/*
buildMessages converts ContextData messages to OpenAI API format
*/
func (prvdr *OpenAIProvider) buildMessages(
	composed *openai.ChatCompletionNewParams,
	ctx aicontext.Context,
) (err error) {
	errnie.Debug("provider.buildMessages")

	msgs, err := ctx.Messages()

	if errnie.Error(err) != nil {
		return err
	}

	messages := make([]openai.ChatCompletionMessageParamUnion, 0, msgs.Len())

	for i := range msgs.Len() {
		msg := msgs.At(i)

		id, err := msg.Id()

		if errnie.Error(err) != nil {
			return err
		}

		role, err := msg.Role()

		if errnie.Error(err) != nil {
			return err
		}

		content, err := msg.Content()

		if errnie.Error(err) != nil {
			return err
		}

		toolCalls, err := msg.ToolCalls()

		if errnie.Error(err) != nil {
			return err
		}

		switch role {
		case "system":
			messages = append(messages, openai.SystemMessage(content))
		case "user":
			messages = append(messages, openai.UserMessage(content))
		case "assistant":
			tcs := make([]openai.ChatCompletionMessageToolCallParam, 0, toolCalls.Len())

			for i := range toolCalls.Len() {
				toolCall := toolCalls.At(i)

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

			errnie.Info("toolCalls", "toolCalls", toolCalls)

			msg := openai.AssistantMessage(content)

			if toolCalls.Len() > 0 {
				msg = openai.ChatCompletionMessageParamUnion{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfString: param.NewOpt(content),
						},
						ToolCalls: tcs,
						Role:      "assistant",
					},
				}
			}

			messages = append(messages, msg)
		case "tool":
			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: param.NewOpt(content),
					},
					ToolCallID: id,
					Role:       "tool",
				},
			})
		default:
			return errnie.Error(errors.New("unknown message role"), "role", role)
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
	tools []mcp.Tool,
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
				Name:        tool.Name,
				Description: param.NewOpt(tool.Description),
				Parameters: openai.FunctionParameters{
					"type":       tool.InputSchema.Type,
					"properties": tool.InputSchema.Properties,
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
	format params.ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	if openaiParams == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

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

	strict := format.Strict()

	openaiParams.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			Type: "json_schema",
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        name,
				Description: param.NewOpt(description),
				Schema:      schema,
				Strict:      param.NewOpt(strict),
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
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("provider.OpenAIEmbedder.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-embedder.ctx.Done():
			errnie.Debug("provider.OpenAIEmbedder.Generate.ctx.Done")
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

			response, err := embedder.client.Embeddings.New(embedder.ctx, openai.EmbeddingNewParams{
				Input:          openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: []string{string(content)}},
				Model:          openai.EmbeddingModelTextEmbeddingAda002,
				Dimensions:     openai.Int(tweaker.GetQdrantDimension()),
				EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
			})
			if err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			if len(response.Data) == 0 {
				out <- datura.New(datura.WithError(errnie.Error(errors.New("no embeddings returned"))))
				return
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

			out <- datura.New(datura.WithPayload(embeddingsBytes))
		}
	}()

	return out
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
