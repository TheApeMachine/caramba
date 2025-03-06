package provider

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"fmt"
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

type ErrorEmptyResponse struct {
	Message string
}

type ErrorNoInputContext struct {
	Message string
}

func (e *ErrorNoInputContext) Error() string {
	return fmt.Sprintf("no input context provided, call Write first: %s", e.Message)
}

func (e *ErrorEmptyResponse) Error() string {
	return fmt.Sprintf("received empty response from OpenAI API: %s", e.Message)
}

type ErrorOpenAIAPI struct {
	Message string
}

func (e *ErrorOpenAIAPI) Error() string {
	return fmt.Sprintf("OpenAI API error: %s", e.Message)
}

type ErrorNoChoices struct {
	Message string
}

func (e *ErrorNoChoices) Error() string {
	return fmt.Sprintf("no choices returned from OpenAI: %s", e.Message)
}

type ErrorUnmarshalingContext struct {
	Message string
}

func (e *ErrorUnmarshalingContext) Error() string {
	return fmt.Sprintf("error unmarshaling context: %s", e.Message)
}

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	*core.BaseComponent
	apiKey   string
	endpoint string
	client   *openai.Client
	params   *ai.Context
	result   *core.Message

	stream *bytes.Buffer
	enc    *gob.Encoder
	dec    *gob.Decoder
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

	buf := new(bytes.Buffer)

	return &OpenAIProvider{
		BaseComponent: core.NewBaseComponent("openai", core.TypeProvider),
		apiKey:        apiKey,
		endpoint:      endpoint,
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
		),
		stream: buf,
		enc:    gob.NewEncoder(buf),
		dec:    gob.NewDecoder(buf),
	}
}

/*
Close cleans up any resources.
*/
func (provider *OpenAIProvider) Close() error {
	// Reset state
	provider.params = nil
	provider.result = nil
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
				Name:        openai.String(tool.Name),
				Description: openai.String(tool.Description),
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

	if params.Process.Name != "" {
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
			Name:        openai.F(params.Process.Name),
			Description: openai.F(params.Process.Description),
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
		errnie.Error("empty response from OpenAI")
		return nil, &ErrorEmptyResponse{Message: "empty response from OpenAI"}
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
			errnie.Error("error streaming chat completion", "error", stream.Err())
			return "", &ErrorOpenAIAPI{Message: fmt.Sprintf("error streaming chat completion: %v", stream.Err())}
		}

		// Check for completed tool calls
		if tool, ok := acc.JustFinishedToolCall(); ok {
			// Parse the arguments string as JSON
			var argsMap map[string]any
			if err := json.Unmarshal([]byte(tool.Arguments), &argsMap); err != nil {
				errnie.Error("failed to parse tool arguments", "error", err)
				// If parsing fails, use the raw string
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
		err := &ErrorOpenAIAPI{Message: "OpenAI API key is not set"}
		errnie.Error(err.Error())
		return "", err
	}

	response, err := provider.client.Chat.Completions.New(ctx, *params)
	if err != nil {
		errnie.Error("error creating chat completion", "error", err)
		return "", &ErrorOpenAIAPI{Message: fmt.Sprintf("error creating chat completion: %v", err)}
	}

	if len(response.Choices) == 0 {
		errnie.Error("no choices returned from OpenAI")
		return "", &ErrorNoChoices{Message: "no choices returned from OpenAI"}
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
			errnie.Error("error marshaling tool calls", "error", err)
			return "", &ErrorOpenAIAPI{Message: fmt.Sprintf("error marshaling tool calls: %v", err)}
		}
		return string(toolCallsJSON), nil
	}

	return "", &ErrorEmptyResponse{Message: "empty response from OpenAI"}
}
