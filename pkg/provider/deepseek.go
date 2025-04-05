package provider

import (
	"context"
	"os"

	"capnproto.org/go/capnp/v3"
	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/mark3labs/mcp-go/mcp"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
DeepseekProvider implements an LLM provider that connects to Deepseek's API.
It supports regular chat completions and streaming responses.
*/
type DeepseekProvider struct {
	client  *deepseek.Client
	pctx    context.Context
	ctx     context.Context
	cancel  context.CancelFunc
	segment *capnp.Segment
}

/*
NewDeepseekProvider creates a new Deepseek provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the DEEPSEEK_API_KEY environment variable.
*/
func NewDeepseekProvider(opts ...DeepseekProviderOption) *DeepseekProvider {
	errnie.Debug("provider.NewDeepseekProvider")

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &DeepseekProvider{
		client: deepseek.NewClient(apiKey),
		pctx:   ctx,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *DeepseekProvider) ID() string {
	return "deepseek"
}

type DeepseekProviderOption func(*DeepseekProvider)

func WithDeepseekAPIKey(apiKey string) DeepseekProviderOption {
	return func(prvdr *DeepseekProvider) {
		prvdr.client = deepseek.NewClient(apiKey)
	}
}

func WithDeepseekEndpoint(endpoint string) DeepseekProviderOption {
	return func(prvdr *DeepseekProvider) {
		// TODO: Implement custom endpoint if the deepseek SDK supports it
	}
}

func (prvdr *DeepseekProvider) Generate(
	params params.Params,
	ctx aicontext.Context,
	tools []mcp.Tool,
) chan *datura.ArtifactBuilder {
	errnie.Debug("provider.DeepseekProvider.Generate")

	out := make(chan *datura.ArtifactBuilder)

	go func() {
		defer close(out)

		model, err := params.Model()
		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		composed := &deepseek.StreamChatCompletionRequest{
			Model:            model,
			Temperature:      float32(params.Temperature()),
			TopP:             float32(params.TopP()),
			PresencePenalty:  float32(params.PresencePenalty()),
			FrequencyPenalty: float32(params.FrequencyPenalty()),
			MaxTokens:        int(params.MaxTokens()),
		}

		if err = prvdr.buildMessages(composed, ctx); err != nil {
			out <- datura.New(datura.WithError(err))
			return
		}

		if err = prvdr.buildTools(composed, tools); err != nil {
			out <- datura.New(datura.WithError(err))
			return
		}

		format, err := params.Format()

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildResponseFormat(composed, format); err != nil {
			out <- datura.New(datura.WithError(err))
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

func (prvdr *DeepseekProvider) Name() string {
	return "deepseek"
}

func (prvdr *DeepseekProvider) handleSingleRequest(
	params *deepseek.StreamChatCompletionRequest,
	channel chan *datura.ArtifactBuilder,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	prms := &deepseek.ChatCompletionRequest{
		Model:       params.Model,
		Messages:    params.Messages,
		Temperature: params.Temperature,
		TopP:        params.TopP,
		MaxTokens:   params.MaxTokens,
	}

	response, err := prvdr.client.CreateChatCompletion(prvdr.ctx, prms)
	if errnie.Error(err) != nil {
		return
	}

	if len(response.Choices) == 0 {
		err = errnie.Error("no response choices")
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

	if err = msg.SetName(string(params.Model)); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	if err = msg.SetContent(response.Choices[0].Message.Content); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	// Create artifact with message content
	channel <- datura.New(
		datura.WithRole(datura.ArtifactRoleAnswer),
		datura.WithScope(datura.ArtifactScopeGeneration),
		datura.WithPayload([]byte(response.Choices[0].Message.Content)),
	)

	return nil
}

func (prvdr *DeepseekProvider) handleStreamingRequest(
	params *deepseek.StreamChatCompletionRequest,
	channel chan *datura.ArtifactBuilder,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream, err := prvdr.client.CreateChatCompletionStream(prvdr.ctx, params)
	if errnie.Error(err) != nil {
		return
	}
	defer stream.Close()

	for {
		response, err := stream.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			errnie.Error("streaming error", "error", err)
			return err
		}

		if len(response.Choices) > 0 {
			content := response.Choices[0].Delta.Content
			if content != "" {
				channel <- datura.New(
					datura.WithRole(datura.ArtifactRoleAnswer),
					datura.WithScope(datura.ArtifactScopeGeneration),
					datura.WithPayload([]byte(content)),
				)
			}
		}
	}

	return nil
}

func (prvdr *DeepseekProvider) buildMessages(
	chatParams *deepseek.StreamChatCompletionRequest,
	ctx aicontext.Context,
) (err error) {
	errnie.Debug("provider.buildMessages")

	msgs, err := ctx.Messages()

	if errnie.Error(err) != nil {
		return err
	}

	messageList := make([]deepseek.ChatCompletionMessage, 0, msgs.Len())

	for i := range msgs.Len() {
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
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleSystem,
				Content: content,
			})
		case "user":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleUser,
				Content: content,
			})
		case "assistant":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleAssistant,
				Content: content,
			})
		default:
			errnie.Error("unknown message role", "role", role)
		}
	}

	chatParams.Messages = messageList

	return nil
}

func (prvdr *DeepseekProvider) buildTools(
	chatParams *deepseek.StreamChatCompletionRequest,
	tools []mcp.Tool,
) (err error) {
	errnie.Debug("provider.buildTools")

	if len(tools) == 0 {
		return nil
	}

	toolList := make([]deepseek.Tool, 0, len(tools))

	for _, tool := range tools {
		properties := make(map[string]interface{})

		for name, prop := range tool.InputSchema.Properties {
			properties[name] = prop
		}

		toolList = append(toolList, deepseek.Tool{
			Type: "function",
			Function: deepseek.Function{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters: &deepseek.FunctionParameters{
					Type:       "object",
					Properties: properties,
					Required:   tool.InputSchema.Required,
				},
			},
		})
	}

	if len(toolList) > 0 {
		chatParams.Tools = toolList
	}

	return nil
}

func (prvdr *DeepseekProvider) buildResponseFormat(
	chatParams *deepseek.StreamChatCompletionRequest,
	format params.ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

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

	// If no format specified, skip
	if name == "" && description == "" && schema == "" {
		return nil
	}

	// Add format instructions as a system message since Deepseek doesn't support direct format control
	formatMsg := deepseek.ChatCompletionMessage{
		Role: deepseek.ChatMessageRoleSystem,
		Content: "Please format your response according to the following schema: " +
			schema + ". " + description,
	}
	chatParams.Messages = append(chatParams.Messages, formatMsg)

	return nil
}

type DeepseekEmbedder struct {
	client *deepseek.Client
	ctx    context.Context
	cancel context.CancelFunc
}

func NewDeepseekEmbedder(opts ...DeepseekEmbedderOption) *DeepseekEmbedder {
	errnie.Debug("provider.NewDeepseekEmbedder")

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	ctx, cancel := context.WithCancel(context.Background())

	client := deepseek.NewClient(apiKey)

	embedder := &DeepseekEmbedder{
		client: client,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(embedder)
	}

	return embedder
}

type DeepseekEmbedderOption func(*DeepseekEmbedder)

func WithDeepseekEmbedderAPIKey(apiKey string) DeepseekEmbedderOption {
	return func(embedder *DeepseekEmbedder) {
		embedder.client = deepseek.NewClient(apiKey)
	}
}

func WithDeepseekEmbedderEndpoint(endpoint string) DeepseekEmbedderOption {
	return func(embedder *DeepseekEmbedder) {
		// TODO: Implement custom endpoint if the deepseek SDK supports it
	}
}

func (embedder *DeepseekEmbedder) Generate(
	buffer chan *datura.ArtifactBuilder,
	fn ...func(artifact *datura.ArtifactBuilder) *datura.ArtifactBuilder,
) chan *datura.ArtifactBuilder {
	errnie.Debug("provider.DeepseekEmbedder.Generate")
	errnie.Warn("Deepseek embedder is not implemented")

	out := make(chan *datura.ArtifactBuilder)
	close(out)
	return out
}
