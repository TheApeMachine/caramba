package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
AnthropicProvider implements an LLM provider that connects to Anthropic's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type AnthropicProvider struct {
	*ProviderData
	client *anthropic.Client
	buffer *stream.Buffer
	ch     chan any
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

	clientOptions := []option.RequestOption{}

	if endpoint != "" {
		clientOptions = append(clientOptions, option.WithBaseURL(endpoint))
	}

	ctx, cancel := context.WithCancel(context.Background())

	p := &AnthropicProvider{
		ProviderData: &ProviderData{
			Params: &ai.ContextData{},
			Result: &core.EventData{},
		},
		client: anthropic.NewClient(
			option.WithAPIKey(apiKey),
		),
		ch:     make(chan any),
		ctx:    ctx,
		cancel: cancel,
	}

	p.buffer = stream.NewBuffer(
		&ai.ContextData{},
		p.ProviderData.Result,
		func(msg any) error {
			var (
				decoded *ai.ContextData
				ok      bool
			)

			if decoded, ok = msg.(*ai.ContextData); !ok {
				errnie.Error("provider.AnthropicProvider.buffer", "msg", msg)
				return errnie.NewErrIO(errnie.NewErrValidation("msg is not an ai.ContextData", "provider", "anthropic"))
			}

			p.ProviderData.Params = decoded
			return nil
		},
	)

	return p
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
func (provider *AnthropicProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicProvider.Write", "p", string(p))

	if n, err = provider.buffer.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = json.Unmarshal(p, provider.ProviderData.Params); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	errnie.Debug("provider.AnthropicProvider.Write", "n", n, "err", err)

	// Create the Anthropic request
	params := anthropic.MessageNewParams{
		Model:     anthropic.F(anthropic.ModelClaude3_5SonnetLatest),
		MaxTokens: anthropic.F(int64(4000)),
		Messages:  anthropic.F(provider.buildMessages(provider.ProviderData.Params)),
	}

	errnie.Debug("provider.AnthropicProvider.Write", "params", params)

	provider.buildTools(provider.ProviderData.Params, &params)
	provider.buildSystemPrompt(provider.ProviderData.Params, &params)

	err = errnie.NewErrIO(provider.handleStreamingRequest(&params))

	return n, err
}

/*
Close cleans up any resources.
*/
func (provider *AnthropicProvider) Close() error {
	errnie.Debug("provider.AnthropicProvider.Close")

	provider.ProviderData.Params = nil
	provider.ProviderData.Result = nil
	provider.buffer = nil

	return nil
}

func (p *AnthropicProvider) buildMessages(
	params *ai.ContextData,
) []anthropic.MessageParam {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "anthropic")
		return nil
	}

	messages := make([]anthropic.MessageParam, 0, len(params.Messages))

	for _, message := range params.Messages {
		switch message.Role {
		case "system":
			// System messages are handled separately in Anthropic SDK
			continue
		case "user":
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(message.Content),
			))
		case "assistant":
			messages = append(messages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(message.Content),
			))
		default:
			errnie.Error("unknown message role", "role", message.Role)
		}
	}

	return messages
}

func (p *AnthropicProvider) buildTools(
	params *ai.ContextData,
	messageParams *anthropic.MessageNewParams,
) {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "anthropic")
		return
	}

	if len(params.Tools) == 0 {
		return
	}

	tools := make([]anthropic.ToolParam, 0, len(params.Tools))

	for _, tool := range params.Tools {
		schema := utils.GenerateSchema[core.Tool]()

		// Create function parameter from tool's schema
		toolParam := anthropic.ToolParam{
			Name:        anthropic.F(tool.ToolData.Name),
			Description: anthropic.F(tool.ToolData.Description),
			InputSchema: anthropic.F(schema), // Assuming this now returns a map or JSON object
		}

		tools = append(tools, toolParam)
	}

	if len(tools) > 0 {
		messageParams.Tools = anthropic.F(tools)
	}
}

func (p *AnthropicProvider) buildSystemPrompt(
	params *ai.ContextData,
	messageParams *anthropic.MessageNewParams,
) {
	errnie.Debug("provider.buildSystemPrompt")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "anthropic")
		return
	}

	// Find system prompt from messages
	var systemPrompt string
	for _, message := range params.Messages {
		if message.Role == "system" {
			systemPrompt = message.Content
			break
		}
	}

	// Add structured output instructions if needed
	if params.Process != nil && params.Process.ProcessData != nil && params.Process.ProcessData.Schema != nil {
		formatInstructions := "Please format your response according to the specified schema: " +
			params.Process.ProcessData.Name + ". " + params.Process.ProcessData.Description

		if systemPrompt != "" {
			systemPrompt = systemPrompt + "\n\n" + formatInstructions
		} else {
			systemPrompt = formatInstructions
		}
	}

	// Set system prompt if we have one
	if systemPrompt != "" {
		messageParams.System = anthropic.F([]anthropic.TextBlockParam{
			anthropic.NewTextBlock(systemPrompt),
		})
	}
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *AnthropicProvider) handleStreamingRequest(
	params *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	ctx := context.Background()

	prvdr.buffer.Stream(ctx, prvdr.ch)

	stream := prvdr.client.Messages.NewStreaming(ctx, *params)
	defer stream.Close()

	errnie.Debug("streaming request initialized")

	count := 0
	accumulatedMessage := anthropic.Message{}

	for stream.Next() {
		event := stream.Current()
		accumulatedMessage.Accumulate(event)

		// Extract text content from deltas
		var content string
		switch delta := event.Delta.(type) {
		case anthropic.ContentBlockDeltaEventDelta:
			content = delta.Text
		}

		if content == "" {
			continue
		}

		errnie.Debug("received stream chunk", "content", content)

		prvdr.Result = core.NewEvent(
			core.NewMessage(
				"assistant",
				"anthropic",
				content,
			),
			nil,
		).EventData

		errnie.Debug("provider.handleStreamingRequest", "result", prvdr.Result)

		select {
		case prvdr.ch <- event:
			errnie.Debug("sent event to channel", "event", event)
		case <-prvdr.ctx.Done():
			return
		default:
			// Don't block if channel is full
		}
	}

	if stream.Err() != nil {
		errnie.Error("streaming error", "error", stream.Err())
		return errnie.NewErrHTTP(stream.Err(), 500)
	}

	errnie.Debug("streaming completed", "chunks", count)
	return nil
}

type AnthropicEmbedderData struct {
	Params *ai.ContextData `json:"params"`
	Result *[]float64      `json:"result"`
}

type AnthropicEmbedder struct {
	*AnthropicEmbedderData
	apiKey   string
	endpoint string
	client   *anthropic.Client
	enc      *json.Encoder
	dec      *json.Decoder
	in       *bufio.ReadWriter
	out      *bufio.ReadWriter
}

func NewAnthropicEmbedder(apiKey string, endpoint string) *AnthropicEmbedder {
	errnie.Debug("provider.NewAnthropicEmbedder")

	in := bufio.NewReadWriter(
		bufio.NewReader(bytes.NewBuffer([]byte{})),
		bufio.NewWriter(bytes.NewBuffer([]byte{})),
	)
	out := bufio.NewReadWriter(
		bufio.NewReader(bytes.NewBuffer([]byte{})),
		bufio.NewWriter(bytes.NewBuffer([]byte{})),
	)

	embedder := &AnthropicEmbedder{
		AnthropicEmbedderData: &AnthropicEmbedderData{},
		apiKey:                apiKey,
		endpoint:              endpoint,
		client:                anthropic.NewClient(option.WithAPIKey(apiKey)),
		enc:                   json.NewEncoder(out),
		dec:                   json.NewDecoder(in),
		in:                    in,
		out:                   out,
	}

	embedder.enc.Encode(embedder.AnthropicEmbedderData)

	return embedder
}

func (embedder *AnthropicEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicEmbedder.Read", "p", string(p))

	if err = embedder.out.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	n, err = embedder.out.Read(p)

	if err != nil {
		errnie.NewErrIO(err)
	}

	return n, err
}

func (embedder *AnthropicEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicEmbedder.Write")

	if n, err = embedder.in.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.in.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.dec.Decode(embedder.AnthropicEmbedderData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.enc.Encode(embedder.AnthropicEmbedderData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	return len(p), nil
}

func (embedder *AnthropicEmbedder) Close() error {
	errnie.Debug("provider.AnthropicEmbedder.Close")

	embedder.AnthropicEmbedderData.Params = nil
	embedder.AnthropicEmbedderData.Result = nil
	return nil
}
