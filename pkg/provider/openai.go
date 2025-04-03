package provider

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"slices"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/system"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	client        *openai.Client
	pctx          context.Context
	ctx           context.Context
	cancel        context.CancelFunc
	status        core.Status
	waiters       []datura.ArtifactScope
	out           chan *datura.Artifact
	paramsBuilder *core.ParamsBuilder
	ctxBuilder    *core.ContextBuilder
	toolset       *tools.Toolset
	protocol      *core.Protocol
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

func (prvdr *OpenAIProvider) Generate(
	buffer chan *datura.Artifact,
	fn ...func(*datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("provider.OpenAIProvider.Generate")

	prvdr.out = make(chan *datura.Artifact, 64)

	go func() {
		defer close(prvdr.out)

		for {
			select {
			case <-prvdr.pctx.Done():
				errnie.Debug("provider.OpenAIProvider.Generate.pctx.Done")
				prvdr.cancel()
				return
			case <-prvdr.ctx.Done():
				errnie.Debug("provider.OpenAIProvider.Generate.ctx.Done")
				return
			case artifact, ok := <-buffer:
				if !ok {
					return
				}

				if prvdr.protocol == nil {
					prvdr.protocol = system.NewHub().GetProtocol(
						datura.GetMetaValue[string](artifact, "protocol"),
					)
				}

				if len(prvdr.waiters) > 0 && !slices.Contains(prvdr.waiters, datura.ArtifactScope(artifact.Scope())) {
					return
				}

				prvdr.waiters = slices.DeleteFunc(prvdr.waiters, func(scope datura.ArtifactScope) bool {
					return scope == datura.ArtifactScope(artifact.Scope())
				})

				var step *datura.Artifact
				step, prvdr.status = prvdr.protocol.HandleMessage(prvdr.ID(), artifact)

				switch prvdr.status {
				case core.StatusWorking:
					prvdr.run(step)
				case core.StatusWaiting:
					prvdr.waiters = append(prvdr.waiters, datura.ArtifactScope(step.Scope()))
				default:
					prvdr.out <- step
				}
			case <-time.After(100 * time.Millisecond):
				// Do nothing
			}
		}
	}()

	return prvdr.out
}

func (prvdr *OpenAIProvider) ID() string {
	return "openai"
}

func (prvdr *OpenAIProvider) run(artifact *datura.Artifact) {
	errnie.Debug("provider.OpenAIProvider.run")

	// Handle different question scopes according to protocol
	switch datura.ArtifactScope(artifact.Scope()) {
	case datura.ArtifactScopeAquire:
		// Initial request from agent, respond with preflight question and acquire ack
		prvdr.out <- datura.New(
			datura.WithRole(datura.ArtifactRoleQuestion),
			datura.WithScope(datura.ArtifactScopePreflight),
			datura.WithMeta("to", datura.GetMetaValue[string](artifact, "from")),
			datura.WithMeta("from", prvdr.ID()),
		)
		prvdr.out <- datura.New(
			datura.WithRole(datura.ArtifactRoleAcknowledge),
			datura.WithScope(datura.ArtifactScopeAquire),
			datura.WithMeta("to", datura.GetMetaValue[string](artifact, "from")),
			datura.WithMeta("from", prvdr.ID()),
		)
		return
	case datura.ArtifactScopePreflight:
		// This shouldn't happen - provider doesn't handle preflight questions
		prvdr.out <- datura.New(
			datura.WithError(errnie.Error("provider received unexpected preflight question")),
		)
		return
	default:
		// Actual generation request
		model, err := prvdr.paramsBuilder.Model()
		if errnie.Error(err) != nil {
			return
		}

		composed := &openai.ChatCompletionNewParams{
			Model:            openai.ChatModel(model),
			Temperature:      openai.Float(prvdr.paramsBuilder.Temperature()),
			TopP:             openai.Float(prvdr.paramsBuilder.TopP()),
			FrequencyPenalty: openai.Float(prvdr.paramsBuilder.FrequencyPenalty()),
			PresencePenalty:  openai.Float(prvdr.paramsBuilder.PresencePenalty()),
		}

		if prvdr.paramsBuilder.MaxTokens() > 1 {
			composed.MaxTokens = openai.Int(int64(prvdr.paramsBuilder.MaxTokens()))
		}

		if err = prvdr.buildMessages(composed); err != nil {
			prvdr.out <- datura.New(
				datura.WithError(err),
			)
			return
		}

		if err = prvdr.buildTools(composed); err != nil {
			prvdr.out <- datura.New(
				datura.WithError(err),
			)
			return
		}

		if prvdr.paramsBuilder.Stream() {
			prvdr.handleStreamingRequest(composed)
		} else {
			prvdr.handleSingleRequest(composed)
		}
	}
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
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	var completion *openai.ChatCompletion

	if completion, err = prvdr.client.Chat.Completions.New(
		prvdr.ctx, *params,
	); errnie.Error(err) != nil {
		return err
	}

	model, err := prvdr.paramsBuilder.Model()

	if errnie.Error(err) != nil {
		return
	}

	msg := core.NewMessageBuilder(
		core.WithRole("assistant"),
		core.WithName(model),
		core.WithContent(completion.Choices[0].Message.Content),
	)

	toolCalls := completion.Choices[0].Message.ToolCalls

	// Abort early if there are no tool calls
	if len(toolCalls) == 0 {
		prvdr.out <- msg.Artifact()
		return nil
	}

	toolCallList, err := core.NewToolCall_List(prvdr.ctxBuilder.Segment(), int32(len(toolCalls)))

	if errnie.Error(err) != nil {
		return err
	}

	for i, toolCall := range toolCalls {
		errnie.Info("toolCall", "tool", toolCall.Function.Name, "id", toolCall.ID)

		if err = toolCallList.At(i).SetId(toolCall.ID); err != nil {
			return errnie.Error(err)
		}

		if err = toolCallList.At(i).SetName(toolCall.Function.Name); err != nil {
			return errnie.Error(err)
		}

		if err = toolCallList.At(i).SetArguments(toolCall.Function.Arguments); err != nil {
			return errnie.Error(err)
		}
	}

	msg.SetToolCalls(toolCallList)
	prvdr.out <- msg.Artifact()

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
			prvdr.out <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(content)),
			)
		}

		if tool, ok := acc.JustFinishedToolCall(); ok {
			params.Messages = append(params.Messages, openai.AssistantMessage(acc.Choices[0].Message.Content))
			prvdr.out <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(tool.Arguments)),
			)
		}

		if refusal, ok := acc.JustFinishedRefusal(); ok && refusal != "" {
			prvdr.out <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(refusal)),
			)
		}

		// Only write non-empty content from chunks
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			prvdr.out <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(chunk.Choices[0].Delta.Content)),
			)
		}
	}

	if err = stream.Err(); err != nil {
		prvdr.out <- datura.New(
			datura.WithError(errnie.Error("Streaming error", "error", err)),
		)
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

	if prvdr.paramsBuilder == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	msgs, err := prvdr.ctxBuilder.Messages()

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
) (err error) {
	errnie.Debug("provider.buildTools")

	if openaiParams == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	if len(prvdr.toolset.Tools) == 0 {
		// No tools, no shoes, no dice.
		return nil
	}

	toolsOut := make([]openai.ChatCompletionToolParam, 0, len(prvdr.toolset.Tools))

	for _, tool := range prvdr.toolset.ToMCP() {
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
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	if openaiParams == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	// Get response format from paramsBuilder
	responseFormat := prvdr.paramsBuilder.ResponseFormat()

	name, err := responseFormat.Name()
	if errnie.Error(err) != nil {
		return err
	}

	description, err := responseFormat.Description()
	if errnie.Error(err) != nil {
		return err
	}

	schema, err := responseFormat.Schema()
	if errnie.Error(err) != nil {
		return err
	}

	strict, err := responseFormat.Strict()
	if errnie.Error(err) != nil {
		return err
	}

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
