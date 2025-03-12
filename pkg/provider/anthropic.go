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
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
AnthropicProvider implements an LLM provider that connects to Anthropic's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type AnthropicProvider struct {
	*ProviderData
	client *anthropic.Client
	buffer *bufio.ReadWriter
	enc    *json.Encoder
	dec    *json.Decoder
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

	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	clientOptions := []option.RequestOption{}
	if endpoint != "" {
		clientOptions = append(clientOptions, option.WithBaseURL(endpoint))
	}

	p := &AnthropicProvider{
		ProviderData: &ProviderData{
			Params: &ai.Context{
				ContextData: &ai.ContextData{
					Messages: []*core.Message{},
				},
			},
			Result: &core.Event{},
		},
		client: anthropic.NewClient(
			option.WithAPIKey(apiKey),
		),
		buffer: buffer,
		enc:    json.NewEncoder(buffer),
		dec:    json.NewDecoder(buffer),
	}

	return p
}

/*
Read implements the io.Reader interface.
*/
func (provider *AnthropicProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicProvider.Read")

	if err = provider.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = provider.buffer.Read(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("provider.AnthropicProvider.Read", "n", n, "err", err)
	return n, err
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
		Messages:  anthropic.F(provider.buildMessages(provider.ProviderData.Params.ContextData)),
	}

	errnie.Debug("provider.AnthropicProvider.Write", "params", params)

	provider.buildTools(provider.ProviderData.Params.ContextData, &params)
	provider.buildSystemPrompt(provider.ProviderData.Params.ContextData, &params)

	err = errnie.NewErrIO(provider.handleStreamingRequest(&params))

	return n, err
}

/*
Close cleans up any resources.
*/
func (provider *AnthropicProvider) Close() error {
	errnie.Debug("provider.AnthropicProvider.Close")

	// Reset state
	provider.ProviderData.Params = nil
	provider.ProviderData.Result = nil

	provider.buffer = nil
	provider.enc = nil
	provider.dec = nil

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
func (provider *AnthropicProvider) handleStreamingRequest(
	params *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	ctx := context.Background()

	stream := provider.client.Messages.NewStreaming(ctx, *params)
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

		provider.Result = core.NewEvent(
			core.NewMessage(
				"assistant",
				"anthropic",
				content,
			),
			nil,
		)

		errnie.Debug("provider.handleStreamingRequest", "result", provider.Result)

		if err = provider.enc.Encode(provider.Result); err != nil {
			errnie.NewErrIO(err)
			return err
		}

		count++
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
