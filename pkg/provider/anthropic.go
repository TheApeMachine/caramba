package provider

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
AnthropicProvider implements an LLM provider that connects to Anthropic's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type AnthropicProvider struct {
	client   *anthropic.Client
	endpoint string
	buffer   *stream.Buffer
	params   *Params
	ctx      context.Context
	cancel   context.CancelFunc
}

/*
NewAnthropicProvider creates a new Anthropic provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the ANTHROPIC_API_KEY environment variable.
*/
func NewAnthropicProvider() *AnthropicProvider {
	errnie.Debug("provider.NewAnthropicProvider")

	ctx, cancel := context.WithCancel(context.Background())
	params := &Params{}

	prvdr := &AnthropicProvider{
		client: anthropic.NewClient(
			option.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		),
		endpoint: "",
		buffer: stream.NewBuffer(func(artfct *datura.Artifact) (err error) {
			errnie.Debug("provider.AnthropicProvider.buffer.fn")
			return errnie.Error(artfct.To(params))
		}),
		params: params,
		ctx:    ctx,
		cancel: cancel,
	}

	return prvdr
}

type AnthropicProviderOption func(*AnthropicProvider)

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

	if n, err = prvdr.buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	composed := anthropic.MessageNewParams{
		Model:       anthropic.F(prvdr.params.Model),
		Temperature: anthropic.F(prvdr.params.Temperature),
		TopP:        anthropic.F(prvdr.params.TopP),
		MaxTokens:   anthropic.F(int64(prvdr.params.MaxTokens)),
	}

	prvdr.buildMessages(&composed)
	prvdr.buildTools(&composed)
	prvdr.buildResponseFormat(&composed)

	if prvdr.params.Stream {
		prvdr.handleStreamingRequest(&composed)
	} else {
		prvdr.handleSingleRequest(&composed)
	}

	return n, nil
}

/*
Close cleans up any resources.
*/
func (provider *AnthropicProvider) Close() error {
	errnie.Debug("provider.AnthropicProvider.Close")
	return nil
}

func (p *AnthropicProvider) handleSingleRequest(
	params *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

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

	if _, err = io.Copy(p.buffer, datura.New(
		datura.WithPayload([]byte(content)),
	)); errnie.Error(err) != nil {
		return err
	}

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

	stream := prvdr.client.Messages.NewStreaming(prvdr.ctx, *params)
	defer stream.Close()

	accumulatedMessage := anthropic.Message{}

	for stream.Next() {
		chunk := stream.Current()
		accumulatedMessage.Accumulate(chunk)

		// Extract text content from deltas
		var content string
		switch delta := chunk.Delta.(type) {
		case anthropic.ContentBlockDeltaEventDelta:
			content = delta.Text
		}

		if content == "" {
			continue
		}

		if _, err = io.Copy(prvdr, datura.New(
			datura.WithPayload([]byte(content)),
		)); errnie.Error(err) != nil {
			continue
		}
	}

	return errnie.Error(stream.Err())
}

func (p *AnthropicProvider) buildMessages(
	messageParams *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.buildMessages")

	if p.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	msgParams := make([]anthropic.MessageParam, 0)

	for _, message := range p.params.Messages {
		switch message.Role {
		case "system":
			messageParams.System = anthropic.F([]anthropic.TextBlockParam{
				anthropic.NewTextBlock(message.Content),
			})
		case "user":
			msgParams = append(msgParams, anthropic.NewUserMessage(
				anthropic.NewTextBlock(message.Content),
			))
		case "assistant":
			msgParams = append(msgParams, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(message.Content),
			))
		default:
			errnie.Error("unknown message role", "role", message.Role)
		}

		messageParams.Messages = anthropic.F(msgParams)
	}

	return nil
}

func (prvdr *AnthropicProvider) buildTools(
	messageParams *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.buildTools")

	toolsOut := make([]anthropic.ToolParam, 0)

	for _, tool := range prvdr.params.Tools {
		properties := make(map[string]interface{})

		for _, property := range tool.Function.Parameters.Properties {
			properties[property.Name] = map[string]interface{}{
				"type":        property.Type,
				"description": property.Description,
				"enum":        property.Enum,
			}
		}

		// Create a map to store the complete schema definition
		schema := map[string]interface{}{
			"type":       "object",
			"properties": properties,
		}

		// Add required fields if present
		if len(tool.Function.Parameters.Required) > 0 {
			schema["required"] = tool.Function.Parameters.Required
		}

		toolParam := anthropic.ToolParam{
			Name:        anthropic.F(tool.Function.Name),
			Description: anthropic.F(tool.Function.Description),
			InputSchema: anthropic.Raw[interface{}](schema),
		}

		toolsOut = append(toolsOut, toolParam)
	}

	messageParams.Tools = anthropic.F(toolsOut)

	return nil
}

func (prvdr *AnthropicProvider) buildResponseFormat(
	messageParams *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	messageParams.Messages.Value = append(
		messageParams.Messages.Value,
		anthropic.NewAssistantMessage(
			anthropic.NewTextBlock(
				strings.Join([]string{
					"Format your response as a JSON object using the following schema.",
					fmt.Sprintf("Schema:\n\n%v", prvdr.params.ResponseFormat.Schema),
					"Strictly follow the schema. Do not leave out required fields, and do not include any non-existent fields or properties.",
					"Output only the JSON object, nothing else, and no Markdown code block.",
				}, "\n\n"),
			),
		),
	)

	return nil
}

type AnthropicEmbedder struct {
	params   *Params
	apiKey   string
	endpoint string
	client   *anthropic.Client
}

func NewAnthropicEmbedder(apiKey string, endpoint string) *AnthropicEmbedder {
	errnie.Debug("provider.NewAnthropicEmbedder")

	return &AnthropicEmbedder{
		params:   &Params{},
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
