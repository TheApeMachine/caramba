package provider

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
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

	apiKey := core.NewConfig().OpenAIAPIKey
	errnie.Debug("provider.NewOpenAIProvider", "apiKey", apiKey)

	ctx, cancel := context.WithCancel(context.Background())
	params := &Params{}

	prvdr := &OpenAIProvider{
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
		),
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
		Model:            openai.F(prvdr.params.Model),
		Temperature:      openai.F(prvdr.params.Temperature),
		TopP:             openai.F(prvdr.params.TopP),
		FrequencyPenalty: openai.F(prvdr.params.FrequencyPenalty),
		PresencePenalty:  openai.F(prvdr.params.PresencePenalty),
	}

	if prvdr.params.MaxTokens > 1 {
		composed.MaxTokens = openai.F(int64(prvdr.params.MaxTokens))
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
		errnie.Info("toolCall", "tool", toolCall.Function.Name)

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

		if content, ok := acc.JustFinishedContent(); ok {
			if _, err = io.Copy(prvdr.buffer, datura.New(
				datura.WithPayload([]byte(content)),
			)); errnie.Error(err) != nil {
				continue
			}
		}

		if tool, ok := acc.JustFinishedToolCall(); ok {
			params.Messages.Value = append(params.Messages.Value, acc.Choices[0].Message)

			switch tool.Name {
			case "browser":
				if _, err = io.Copy(prvdr.buffer, datura.New(
					datura.WithPayload([]byte(tool.JSON.RawJSON())),
				)); errnie.Error(err) != nil {
					continue
				}
			}
		}

		if refusal, ok := acc.JustFinishedRefusal(); ok {
			if _, err = io.Copy(prvdr.buffer, datura.New(
				datura.WithPayload([]byte(refusal)),
			)); errnie.Error(err) != nil {
				continue
			}
		}

		// It's best to use chunks after handling JustFinished events
		if len(chunk.Choices) > 0 {
			if _, err = io.Copy(prvdr.buffer, datura.New(
				datura.WithPayload([]byte(chunk.Choices[0].Delta.Content)),
			)); errnie.Error(err) != nil {
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
					ID:   openai.F(toolCall.ID),
					Type: openai.F(openai.ChatCompletionMessageToolCallTypeFunction),
					Function: openai.F(openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      openai.F(toolCall.Function.Name),
						Arguments: openai.F(toolCall.Function.Arguments),
					}),
				})
			}

			msg := openai.ChatCompletionAssistantMessageParam{
				Role: openai.F(openai.ChatCompletionAssistantMessageParamRoleAssistant),
				Name: openai.F(message.Name),
				Content: openai.F([]openai.ChatCompletionAssistantMessageParamContentUnion{
					openai.ChatCompletionAssistantMessageParamContent{
						Type: openai.F(openai.ChatCompletionAssistantMessageParamContentTypeText),
						Text: openai.F(message.Content),
					},
				}),
			}

			if len(toolCalls) > 0 {
				msg.ToolCalls = openai.F(toolCalls)
			}

			messages = append(messages, msg)
		case "tool":
			messages = append(messages, openai.ToolMessage(message.Reference, message.Content))
		default:
			fmt.Println(message.Role, message.Content)
			return errnie.Error(errors.New("unknown message role"))
		}
	}

	composed.Messages = openai.F(messages)

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
	params   *Params
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
		params:   &Params{},
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
