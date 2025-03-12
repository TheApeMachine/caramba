package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

type ProviderData struct {
	Params *ai.ContextData `json:"params"`
	Result *core.Event     `json:"result"`
}

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	*ProviderData
	client *openai.Client
	buffer *bufio.ReadWriter
	enc    *json.Encoder
	dec    *json.Decoder
}

/*
NewOpenAIProvider creates a new OpenAI provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
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

	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	p := &OpenAIProvider{
		ProviderData: &ProviderData{
			Params: &ai.ContextData{
				Messages: []*core.Message{},
			},
			Result: &core.Event{},
		},
		client: openai.NewClient(
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
func (provider *OpenAIProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIProvider.Read")

	if err = provider.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = provider.buffer.Read(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("provider.OpenAIProvider.Read", "n", n, "err", err)
	return n, err
}

/*
Write implements the io.Writer interface.
*/
func (provider *OpenAIProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIProvider.Write", "p", string(p))

	if n, err = provider.buffer.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = json.Unmarshal(p, provider.ProviderData.Params); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	errnie.Debug("provider.OpenAIProvider.Write", "n", n, "err", err)

	// Create the OpenAI request
	openaiParams := openai.ChatCompletionNewParams{
		Model:    openai.F(openai.ChatModelGPT4o),
		Messages: openai.F(provider.buildMessages(provider.ProviderData.Params)),
	}

	errnie.Debug("provider.OpenAIProvider.Write", "openaiParams", openaiParams)

	provider.buildTools(provider.ProviderData.Params, &openaiParams)
	provider.buildResponseFormat(provider.ProviderData.Params, &openaiParams)

	err = errnie.NewErrIO(provider.handleStreamingRequest(&openaiParams))

	return n, err
}

/*
Close cleans up any resources.
*/
func (provider *OpenAIProvider) Close() error {
	errnie.Debug("provider.OpenAIProvider.Close")

	// Reset state
	provider.ProviderData.Params = nil
	provider.ProviderData.Result = nil

	provider.buffer = nil
	provider.enc = nil
	provider.dec = nil

	return nil
}

func (p *OpenAIProvider) buildMessages(
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
func (provider *OpenAIProvider) handleStreamingRequest(
	params *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	ctx := context.Background()

	stream := provider.client.Chat.Completions.NewStreaming(ctx, *params)
	acc := openai.ChatCompletionAccumulator{}
	defer stream.Close()

	errnie.Debug("streaming request initialized")

	count := 0

	for stream.Next() {
		currentChunk := stream.Current()
		errnie.Debug("received stream chunk",
			"chunk_id", currentChunk.ID,
			"content", currentChunk.Choices[0].Delta.Content,
		)
		acc.AddChunk(currentChunk)

		errnie.Debug("building event", "content", currentChunk.Choices[0].Delta.Content)

		provider.Result = core.NewEvent(
			core.NewMessage(
				"assistant",
				"openai",
				currentChunk.Choices[0].Delta.Content,
			),
			nil,
		)

		errnie.Debug("provider.handleStreamingRequest", "result", provider.Result)

		if err = provider.enc.Encode(provider.Result); err != nil {
			errnie.NewErrIO(err)
			return
		}

		if stream.Err() != nil {
			errnie.Error("stream error",
				"error", stream.Err(),
			)
			err = errnie.NewErrHTTP(stream.Err(), 500)
			return
		}

		count++
	}

	if err != nil {
		errnie.Error("streaming failed",
			"error", err,
			"chunks", count,
		)
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
