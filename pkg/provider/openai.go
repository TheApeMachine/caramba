package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

type ProviderEvent struct {
	Message  *core.Message  `json:"message"`
	ToolCall *core.ToolCall `json:"tool_call"`
	Error    error          `json:"error"`
}

type ProviderData struct {
	Params *ai.Context    `json:"params"`
	Result *ProviderEvent `json:"result"`
}

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	*ProviderData
	apiKey   string
	endpoint string
	client   *openai.Client
	enc      *json.Encoder
	dec      *json.Decoder
	in       *bytes.Buffer
	out      *bytes.Buffer
}

/*
NewOpenAIProvider creates a new OpenAI provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
*/
func NewOpenAIProvider(
	apiKey string, endpoint string,
) *OpenAIProvider {
	errnie.Debug("NewOpenAIProvider", "package", "provider")

	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	if endpoint == "" {
		endpoint = viper.GetViper().GetString("endpoints.openai")
	}

	in := new(bytes.Buffer)
	out := new(bytes.Buffer)

	p := &OpenAIProvider{
		apiKey:   apiKey,
		endpoint: endpoint,
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
		),
		enc: json.NewEncoder(out),
		dec: json.NewDecoder(in),
		in:  in,
		out: out,
	}

	// Pre-encode the provider data to JSON for reading
	p.enc.Encode(p.ProviderData)

	return p
}

/*
Read implements the io.Reader interface.
*/
func (provider *OpenAIProvider) Read(p []byte) (n int, err error) {
	if provider.out.Len() == 0 {
		if err = errnie.NewErrIO(provider.enc.Encode(provider.ProviderData)); err != nil {
			return 0, err
		}
	}

	return provider.out.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (provider *OpenAIProvider) Write(p []byte) (n int, err error) {
	// Reset the output buffer whenever we write new data
	if provider.out.Len() > 0 {
		provider.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = provider.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf ProviderData
	if decErr := provider.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		provider.ProviderData.Params = buf.Params

		// Re-encode to the output buffer for subsequent reads
		if encErr := provider.enc.Encode(provider.ProviderData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

/*
Close cleans up any resources.
*/
func (provider *OpenAIProvider) Close() error {
	// Reset state
	provider.ProviderData.Params = nil
	provider.ProviderData.Result = nil
	return nil
}

func (p *OpenAIProvider) buildMessages(
	params *ai.Context,
) []openai.ChatCompletionMessageParamUnion {
	errnie.Debug("buildMessages", "package", "provider")

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
	params *ai.Context,
	openaiParams *openai.ChatCompletionNewParams,
) []openai.ChatCompletionToolParam {
	errnie.Debug("buildTools", "package", "provider")

	toolsOut := make([]openai.ChatCompletionToolParam, 0, len(params.Tools))

	for _, tool := range params.Tools {
		schema := utils.GenerateSchema[core.Tool]()

		// Create function parameter from tool's schema
		toolParam := openai.ChatCompletionToolParam{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(openai.FunctionDefinitionParam{
				Name:        openai.String(tool.Name()),
				Description: openai.String(tool.Description()),
				Parameters:  openai.F(schema.(openai.FunctionParameters)), // Type assertion to FunctionParameters
			}),
		}

		toolsOut = append(toolsOut, toolParam)
	}
	openaiParams.Tools = openai.F(toolsOut)

	return toolsOut
}

func (p *OpenAIProvider) buildResponseFormat(
	params *ai.Context,
	openaiParams *openai.ChatCompletionNewParams,
) {
	errnie.Debug("buildResponseFormat", "package", "provider")

	if params.Process.ProcessData.Schema != nil {
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
}

func (p *OpenAIProvider) GenerateResponse(ctx *ai.Context) (*core.Message, error) {
	errnie.Debug("GenerateResponse", "provider", "openai", "model", ctx.Model)

	openaiParams := openai.ChatCompletionNewParams{
		Model:    openai.F(openai.ChatModelGPT4o),
		Messages: openai.F(p.buildMessages(ctx)),
	}

	openaiParams.Tools = openai.F(p.buildTools(ctx, &openaiParams))
	p.buildResponseFormat(ctx, &openaiParams)

	var response string
	var err error

	if ctx.Stream {
		response, err = p.handleStreamingRequest(&openaiParams)
	} else {
		response, err = p.handleStandardRequest(&openaiParams)
	}

	if err != nil {
		errnie.Error("failed to generate response", "error", err)
		return nil, err
	}

	if response == "" {
		return nil, errnie.NewErrHTTP(errors.New("empty response from OpenAI"), 400)
	}

	return core.NewMessage("assistant", "openai", response), nil
}

/*
handleStreamingRequest processes a streaming completion request
and accumulates the chunks into a single response.
*/
func (provider *OpenAIProvider) handleStreamingRequest(params *openai.ChatCompletionNewParams) (string, error) {
	errnie.Debug("handleStreamingRequest", "package", "provider")

	ctx := context.Background()

	stream := provider.client.Chat.Completions.NewStreaming(ctx, *params)
	acc := openai.ChatCompletionAccumulator{}
	defer stream.Close()

	var content string

	for stream.Next() {
		chunk := stream.Current()
		acc.AddChunk(chunk)

		// Check for completed content
		if completedContent, ok := acc.JustFinishedContent(); ok && completedContent != "" {
			content = completedContent
		}

		if stream.Err() != nil {
			return "", errnie.NewErrHTTP(stream.Err(), 500)
		}

		// Check for completed tool calls
		if tool, ok := acc.JustFinishedToolCall(); ok {
			// Parse the arguments string as JSON
			var argsMap map[string]any
			if err := json.Unmarshal([]byte(tool.Arguments), &argsMap); err != nil {
				errnie.NewErrParse(err)

				argsMap = map[string]any{
					"raw_args": tool.Arguments,
				}
			}

			content = tool.Arguments
		}
	}

	return content, nil
}

/*
handleStandardRequest processes a standard (non-streaming) completion request.
*/
func (provider *OpenAIProvider) handleStandardRequest(params *openai.ChatCompletionNewParams) (string, error) {
	errnie.Debug("handleStandardRequest", "package", "provider")

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Log basic request information
	errnie.Debug(
		"Processing standard request",
		"package", "provider",
		"model", params.Model,
		"messages", len(params.Messages.Value),
		"temperature", params.Temperature,
		"max_tokens", params.MaxTokens,
	)

	// Check API key and show clear error
	if provider.apiKey == "" {
		return "", errnie.NewErrValidation("OpenAI API key is not set")
	}

	response, err := provider.client.Chat.Completions.New(ctx, *params)
	if err != nil {
		return "", errnie.NewErrHTTP(err, 500)
	}

	if len(response.Choices) == 0 {
		return "", errnie.NewErrValidation("no choices returned from OpenAI")
	}

	choice := response.Choices[0]

	// Handle standard text response
	if choice.Message.Content != "" {
		return choice.Message.Content, nil
	}

	// Handle tool calls response
	if len(choice.Message.ToolCalls) > 0 {
		toolCallsJSON, err := json.Marshal(choice.Message.ToolCalls)
		if err != nil {
			return "", errnie.NewErrParse(err)
		}
		return string(toolCallsJSON), nil
	}

	return "", errnie.NewErrHTTP(errors.New("empty response from OpenAI"), 400)
}
