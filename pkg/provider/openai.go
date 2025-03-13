package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

type ProviderData struct {
	Params *ai.ContextData `json:"params"`
	Result *core.EventData `json:"result"`
}

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	*ProviderData
	client *openai.Client
	*stream.Buffer
	ch     chan any
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewOpenAIProvider creates a new OpenAI provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIProvider(
	apiKey string,
	endpoint string,
) *OpenAIProvider {
	errnie.Debug("provider.NewOpenAIProvider")

	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	if endpoint == "" {
		endpoint = viper.GetViper().GetString("endpoints.openai")
	}

	ctx, cancel := context.WithCancel(context.Background())

	provider := &OpenAIProvider{
		ProviderData: &ProviderData{
			Params: &ai.ContextData{},
			Result: &core.EventData{},
		},
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
		),
		ch:     make(chan any, 128),
		ctx:    ctx,
		cancel: cancel,
	}

	// Create buffer with handler that processes OpenAI API requests
	provider.Buffer = stream.NewBuffer(
		provider.Params,        // Receiver: we'll get ContextData
		provider.Result,        // Sender: we'll send Event
		provider.processParams, // Handler: convert and call OpenAI
	)

	return provider
}

// processParams is the handler function that processes incoming parameters
func (provider *OpenAIProvider) processParams(params any) error {
	provider.Params = params.(*ai.ContextData)

	// Debug the parameters
	errnie.Debug(
		"provider.processParams",
		"model", provider.Params.Model,
		"messages", len(provider.Params.Messages),
	)

	// Create OpenAI API parameters
	openaiParams := openai.ChatCompletionNewParams{
		Model:    openai.F(provider.Params.Model),
		Messages: openai.F(provider.buildMessages(provider.Params)),
	}

	// Set optional parameters
	if provider.Params.Temperature > 0 {
		openaiParams.Temperature = openai.F(provider.Params.Temperature)
	}
	if provider.Params.TopP > 0 {
		openaiParams.TopP = openai.F(provider.Params.TopP)
	}
	if provider.Params.MaxTokens > 0 {
		openaiParams.MaxTokens = openai.F(int64(provider.Params.MaxTokens))
	}
	if provider.Params.PresencePenalty != 0 {
		openaiParams.PresencePenalty = openai.F(provider.Params.PresencePenalty)
	}
	if provider.Params.FrequencyPenalty != 0 {
		openaiParams.FrequencyPenalty = openai.F(provider.Params.FrequencyPenalty)
	}

	// Add tools if any
	if len(provider.Params.Tools) > 0 {
		provider.buildTools(provider.Params, &openaiParams)
	}

	// Add response format if needed
	if provider.Params.Process != nil {
		provider.buildResponseFormat(provider.Params, &openaiParams)
	}

	// Handle streaming vs non-streaming
	var err error
	if provider.Params.Stream {
		err = provider.handleStreamingRequest(&openaiParams)
	} else {
		err = provider.handleSingleRequest(&openaiParams)
	}

	if err != nil {
		errnie.Error("OpenAI request failed", "error", err)
		// Update the Result with the error
		provider.Result = core.NewEvent(
			core.NewMessage("assistant", "system", "Error processing request"),
			err,
		).EventData
		return err
	}

	return nil
}

/*
Read implements the io.Reader interface.
*/
func (provider *OpenAIProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIProvider.Read")
	return provider.Buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (provider *OpenAIProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIProvider.Write", "p", string(p))
	return provider.Buffer.Write(p)
}

/*
Close cleans up any resources.
*/
func (provider *OpenAIProvider) Close() error {
	errnie.Debug("provider.OpenAIProvider.Close")
	return provider.Buffer.Close()
}

// buildMessages converts ContextData messages to OpenAI API format
func (provider *OpenAIProvider) buildMessages(
	params *ai.ContextData,
) []openai.ChatCompletionMessageParamUnion {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "openai")
		return nil
	}

	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(params.Messages))

	for _, message := range params.Messages {
		switch message.Role {
		case "system":
			messages = append(messages, openai.SystemMessage(message.Content))
		case "user":
			messages = append(messages, openai.UserMessage(message.Content))
		case "assistant":
			messages = append(messages, openai.AssistantMessage(message.Content))
		default:
			errnie.Error("unknown message role", "role", message.Role)
		}
	}

	return messages
}

// handleSingleRequest processes a single (non-streaming) completion request
func (provider *OpenAIProvider) handleSingleRequest(
	params *openai.ChatCompletionNewParams,
) error {
	errnie.Debug("provider.handleSingleRequest")

	ctx := context.Background()

	// Make the API call
	completion, err := provider.client.Chat.Completions.New(ctx, *params)
	if err != nil {
		return err
	}

	// Update the Result with the response
	if len(completion.Choices) > 0 {
		content := completion.Choices[0].Message.Content
		provider.Result = core.NewEvent(
			core.NewMessage("assistant", "openai", content),
			nil,
		).EventData

		// Encode and write the result to the buffer for reading
		buf := bytes.NewBuffer([]byte{})
		if err := gob.NewEncoder(buf).Encode(provider.ProviderData.Result); err != nil {
			errnie.Error("failed to encode result", "error", err)
			return err
		}
		provider.Buffer.Write(buf.Bytes())
		errnie.Debug("provider.handleSingleRequest", "response", content)
	}

	return nil
}

func (p *OpenAIProvider) buildTools(
	params *ai.ContextData,
	openaiParams *openai.ChatCompletionNewParams,
) []openai.ChatCompletionToolParam {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "openai")
		return nil
	}

	toolsOut := make([]openai.ChatCompletionToolParam, 0, len(params.Tools))

	for _, tool := range params.Tools {
		schema := utils.GenerateSchema[core.Tool]()

		// Create function parameter from tool's schema
		toolParam := openai.ChatCompletionToolParam{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(openai.FunctionDefinitionParam{
				Name:        openai.String(tool.ToolData.Name),
				Description: openai.String(tool.ToolData.Description),
				Parameters:  openai.F(schema.(openai.FunctionParameters)), // Type assertion to FunctionParameters
			}),
		}

		toolsOut = append(toolsOut, toolParam)
	}
	openaiParams.Tools = openai.F(toolsOut)

	return toolsOut
}

func (p *OpenAIProvider) buildResponseFormat(
	params *ai.ContextData,
	openaiParams *openai.ChatCompletionNewParams,
) {
	errnie.Debug("provider.buildResponseFormat")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "openai")
		return
	}

	if params.Process == nil || params.Process.ProcessData == nil || params.Process.ProcessData.Schema == nil {
		return
	}

	// Convert the schema to a string representation for OpenAI
	schemaJSON, err := json.Marshal(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"message": map[string]any{
				"type": "string",
			},
		},
	})
	if err != nil {
		errnie.Error("failed to convert schema to JSON", "error", err)
		return
	}

	// Parse the schema JSON back into a generic any for OpenAI
	var schemaObj any
	if err := json.Unmarshal(schemaJSON, &schemaObj); err != nil {
		errnie.Error("failed to parse schema JSON", "error", err)
		return
	}

	schemaParam := openai.ResponseFormatJSONSchemaJSONSchemaParam{
		Name:        openai.F(params.Process.ProcessData.Name),
		Description: openai.F(params.Process.ProcessData.Description),
		Schema:      openai.F(schemaObj),
		Strict:      openai.Bool(true),
	}

	openaiParams.ResponseFormat = openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](
		openai.ResponseFormatJSONSchemaParam{
			Type:       openai.F(openai.ResponseFormatJSONSchemaTypeJSONSchema),
			JSONSchema: openai.F(schemaParam),
		},
	)
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *OpenAIProvider) handleStreamingRequest(
	params *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")
	defer close(prvdr.ch)
	defer prvdr.cancel()

	prvdr.Buffer.Stream(prvdr.ctx, prvdr.ch)

	stream := prvdr.client.Chat.Completions.NewStreaming(prvdr.ctx, *params)
	acc := openai.ChatCompletionAccumulator{}
	defer stream.Close()

	errnie.Debug("streaming request initialized")

	for stream.Next() {
		chunk := stream.Current()
		event := core.NewEvent(
			core.NewMessage(
				"assistant",
				prvdr.Params.Model,
				"",
			),
			nil,
		)

		if ok := acc.AddChunk(chunk); !ok {
			errnie.Error("chunk dropped", "id", acc.ID)
		}

		// When this fires, the current chunk value will not contain content data
		if content, ok := acc.JustFinishedContent(); ok {
			event.Message.Content = content
			errnie.Debug("received content chunk", "content", content)
		}

		if tool, ok := acc.JustFinishedToolCall(); ok {
			event.ToolCalls = append(event.ToolCalls, core.NewToolCall(
				acc.Choices[tool.Index].Message.ToolCalls[tool.Index].ID,
				acc.Choices[tool.Index].Message.ToolCalls[tool.Index].Function.Name,
				map[string]any{
					"arguments": acc.Choices[tool.Index].Message.ToolCalls[tool.Index].Function.Arguments,
				},
			))
		}

		if refusal, ok := acc.JustFinishedRefusal(); ok {
			event.Error = errnie.NewErrValidation(refusal, "provider", "openai").Error()
		}

		// Update the message content in the existing Result object
		prvdr.Result.Message.Content = acc.Choices[0].Message.Content

		// We'll update our main Result object
		prvdr.ProviderData.Result = event.EventData

		// Still send to channel for compatibility
		select {
		case prvdr.ch <- event:
			errnie.Debug("sent event to channel", "event", event)
		case <-prvdr.ctx.Done():
			return
		default:
			// Don't block if channel is full
		}
	}

	if err = stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		errnie.NewErrIO(err)
		return
	}

	return err
}

type OpenAIEmbedderData struct {
	Params *ai.ContextData `json:"params"`
	Result *[]float64      `json:"result"`
}

type OpenAIEmbedder struct {
	*OpenAIEmbedderData
	apiKey   string
	endpoint string
	client   *openai.Client
	enc      *json.Encoder
	dec      *json.Decoder
	in       *bufio.ReadWriter
	out      *bufio.ReadWriter
}

func NewOpenAIEmbedder(apiKey string, endpoint string) *OpenAIEmbedder {
	errnie.Debug("provider.NewOpenAIEmbedder")

	in := bufio.NewReadWriter(
		bufio.NewReader(bytes.NewBuffer([]byte{})),
		bufio.NewWriter(bytes.NewBuffer([]byte{})),
	)
	out := bufio.NewReadWriter(
		bufio.NewReader(bytes.NewBuffer([]byte{})),
		bufio.NewWriter(bytes.NewBuffer([]byte{})),
	)

	embedder := &OpenAIEmbedder{
		OpenAIEmbedderData: &OpenAIEmbedderData{},
		apiKey:             apiKey,
		endpoint:           endpoint,
		client:             openai.NewClient(option.WithAPIKey(apiKey)),
		enc:                json.NewEncoder(out),
		dec:                json.NewDecoder(in),
		in:                 in,
		out:                out,
	}

	embedder.enc.Encode(embedder.OpenAIEmbedderData)

	return embedder
}

func (embedder *OpenAIEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIEmbedder.Read", "p", string(p))

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

func (embedder *OpenAIEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIEmbedder.Write")

	if n, err = embedder.in.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.in.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.dec.Decode(embedder.OpenAIEmbedderData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.enc.Encode(embedder.OpenAIEmbedderData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	return len(p), nil
}

func (embedder *OpenAIEmbedder) Close() error {
	errnie.Debug("provider.OpenAIEmbedder.Close")

	embedder.OpenAIEmbedderData.Params = nil
	embedder.OpenAIEmbedderData.Result = nil
	return nil
}
