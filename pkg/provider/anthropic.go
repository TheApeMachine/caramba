package provider

import (
	"context"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/spf13/viper"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
AnthropicProvider implements an LLM provider that connects to Anthropic's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type AnthropicProvider struct {
	client *anthropic.Client
	buffer *stream.Buffer
	params *aiCtx.Artifact
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewAnthropicProvider creates a new Anthropic provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the ANTHROPIC_API_KEY environment variable.
*/
func NewAnthropicProvider(
	apiKey string,
	endpoint string,
) *AnthropicProvider {
	errnie.Debug("provider.NewAnthropicProvider")

	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}

	if endpoint == "" {
		endpoint = viper.GetViper().GetString("endpoints.anthropic")
	}

	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &AnthropicProvider{
		client: anthropic.NewClient(
			option.WithAPIKey(apiKey),
		),
		params: aiCtx.New(
			anthropic.ModelClaude3_5SonnetLatest,
			nil,
			nil,
			nil,
			0.7,
			1.0,
			0,
			0.0,
			0.0,
			2048,
			false,
		),
		ctx:    ctx,
		cancel: cancel,
	}

	prvdr.buffer = stream.NewBuffer(
		func(event *event.Artifact) error {
			errnie.Debug("provider.AnthropicProvider.buffer.fn", "event", event)

			payload, err := event.Payload()

			if errnie.Error(err) != nil {
				return err
			}

			_, err = prvdr.params.Write(payload)

			if errnie.Error(err) != nil {
				return err
			}

			return nil
		},
	)

	return prvdr
}

/*
Read implements the io.Reader interface.
*/
func (provider *AnthropicProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicProvider.Read")
	return provider.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (prvdr *AnthropicProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicProvider.Write")

	n, err = prvdr.buffer.Write(p)

	if errnie.Error(err) != nil {
		return n, err
	}

	composed := anthropic.MessageNewParams{}

	model, err := prvdr.params.Model()
	if err != nil {
		errnie.Error("failed to get model", "error", err)
		return n, err
	}

	composed.Model = anthropic.F(model)

	prvdr.buildMessages(prvdr.params, &composed)
	prvdr.buildTools(prvdr.params, &composed)

	if prvdr.params.Stream() {
		prvdr.handleStreamingRequest(&composed)
	} else {
		prvdr.handleSingleRequest(&composed)
	}

	return len(p), nil
}

/*
Close cleans up any resources.
*/
func (provider *AnthropicProvider) Close() error {
	errnie.Debug("provider.AnthropicProvider.Close")
	return nil
}

func (p *AnthropicProvider) buildMessages(
	params *aiCtx.Artifact,
	messageParams *anthropic.MessageNewParams,
) *anthropic.MessageNewParams {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "anthropic")
		return nil
	}

	messages, err := params.Messages()
	if err != nil {
		errnie.Error("failed to get messages", "error", err)
		return nil
	}

	msgParams := make([]anthropic.MessageParam, 0)

	for idx := range messages.Len() {
		message := messages.At(idx)

		role, err := message.Role()
		if err != nil {
			errnie.Error("failed to get message role", "error", err)
			continue
		}

		content, err := message.Content()
		if err != nil {
			errnie.Error("failed to get message content", "error", err)
			continue
		}

		switch role {
		case "system":
			messageParams.System = anthropic.F([]anthropic.TextBlockParam{
				anthropic.NewTextBlock(content),
			})
		case "user":
			msgParams = append(msgParams, anthropic.NewUserMessage(
				anthropic.NewTextBlock(content),
			))
		case "assistant":
			msgParams = append(msgParams, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(content),
			))
		default:
			errnie.Error("unknown message role", "role", role)
		}

		messageParams.Messages = anthropic.F(msgParams)
	}

	return messageParams
}

func (p *AnthropicProvider) buildTools(
	params *aiCtx.Artifact,
	messageParams *anthropic.MessageNewParams,
) {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "anthropic")
		return
	}

	tools, err := params.Tools()
	if err != nil {
		errnie.Error("failed to get tools", "error", err)
		return
	}

	toolList := make([]anthropic.ToolParam, 0, tools.Len())

	for idx := range tools.Len() {
		tool := tools.At(idx)

		schema := utils.GenerateSchema[struct{}]()

		name, err := tool.Name()
		if err != nil {
			errnie.Error("failed to get tool name", "error", err)
			continue
		}

		description, err := tool.Description()
		if err != nil {
			errnie.Error("failed to get tool description", "error", err)
			continue
		}

		// Create function parameter from tool's schema
		toolParam := anthropic.ToolParam{
			Name:        anthropic.String(name),
			Description: anthropic.String(description),
			InputSchema: anthropic.F(schema), // Assuming this now returns a map or JSON object
		}

		toolList = append(toolList, toolParam)
	}

	if len(toolList) > 0 {
		messageParams.Tools = anthropic.F(toolList)
	}
}

func (p *AnthropicProvider) handleSingleRequest(
	params *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	go func() {
		defer close(p.buffer.Stream)

		response, err := p.client.Messages.New(p.ctx, *params)
		if errnie.Error(err) != nil {
			return
		}

		msg := response.Content
		if msg == nil {
			errnie.Error("failed to get message", "error", err)
			return
		}

		content := msg[0].Text

		m, err := message.New(
			message.AssistantRole,
			"",
			content,
		).Message().Marshal()

		if errnie.Error(err) != nil {
			return
		}

		p.buffer.Stream <- event.New(
			"provider.anthropic",
			event.MessageEvent,
			event.AssistantRole,
			m,
		)
	}()

	return nil
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *AnthropicProvider) handleStreamingRequest(
	params *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	go func() {
		defer close(prvdr.buffer.Stream)

		stream := prvdr.client.Messages.NewStreaming(prvdr.ctx, *params)
		defer stream.Close()

		accumulatedMessage := anthropic.Message{}

		for stream.Next() {
			evt := stream.Current()
			accumulatedMessage.Accumulate(evt)

			// Extract text content from deltas
			var content string
			switch delta := evt.Delta.(type) {
			case anthropic.ContentBlockDeltaEventDelta:
				content = delta.Text
			}

			if content == "" {
				continue
			}

			msg, err := message.New(
				message.AssistantRole,
				"",
				content,
			).Message().Marshal()

			if errnie.Error(err) != nil {
				continue
			}

			prvdr.buffer.Stream <- event.New(
				"provider.anthropic",
				event.MessageEvent,
				event.AssistantRole,
				msg,
			)
		}

		errnie.Error(stream.Err())
	}()

	return nil
}

type AnthropicEmbedder struct {
	params   *aiCtx.Artifact
	apiKey   string
	endpoint string
	client   *anthropic.Client
}

func NewAnthropicEmbedder(apiKey string, endpoint string) *AnthropicEmbedder {
	errnie.Debug("provider.NewAnthropicEmbedder")

	return &AnthropicEmbedder{
		params:   &aiCtx.Artifact{},
		apiKey:   apiKey,
		endpoint: endpoint,
		client:   anthropic.NewClient(option.WithAPIKey(apiKey)),
	}
}

func (embedder *AnthropicEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicEmbedder.Read", "p", string(p))
	return 0, nil
}

func (embedder *AnthropicEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicEmbedder.Write")
	return len(p), nil
}

func (embedder *AnthropicEmbedder) Close() error {
	errnie.Debug("provider.AnthropicEmbedder.Close")

	embedder.params = nil
	return nil
}
