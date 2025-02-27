package llm

import (
	"context"
	"encoding/json"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/agent/core"
)

// OpenAIProvider implements the LLMProvider interface for OpenAI
type OpenAIProvider struct {
	APIKey  string
	Model   string
	BaseURL string
	Client  *openai.Client
}

// NewOpenAIProvider creates a new OpenAI provider
func NewOpenAIProvider(apiKey string, model string) *OpenAIProvider {
	baseURL := viper.GetString("endpoints.openai")
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	return &OpenAIProvider{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: baseURL,
		Client:  openai.NewClient(option.WithAPIKey(apiKey)),
	}
}

// Name returns the name of the LLM provider
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// GenerateResponse generates a response from the LLM
func (p *OpenAIProvider) GenerateResponse(
	ctx context.Context,
	params core.LLMParams,
) core.LLMResponse {
	openaiParams := openai.ChatCompletionNewParams{
		Model:    openai.F(openai.ChatModelGPT4o),
		Messages: openai.F(p.buildMessages(params)),
	}

	if tools := p.buildTools(params, &openaiParams); len(tools) > 0 {
		openaiParams.Tools = openai.F(tools)
	}

	p.buildResponseFormat(params, &openaiParams)

	completion, err := p.Client.Chat.Completions.New(ctx, openaiParams)
	if err != nil {
		return core.LLMResponse{
			Error: err,
		}
	}

	out := core.LLMResponse{}

	if len(completion.Choices) > 0 {
		out.Content = completion.Choices[0].Message.Content

		if len(completion.Choices[0].Message.ToolCalls) > 0 {
			for _, toolCall := range completion.Choices[0].Message.ToolCalls {
				out.ToolCalls = append(out.ToolCalls, core.ToolCall{
					Name: toolCall.Function.Name,
					Args: map[string]interface{}{
						"args": toolCall.Function.Arguments,
					},
				})
			}
		}
	}

	return out
}

// StreamResponse generates a response from the LLM and streams it
func (p *OpenAIProvider) StreamResponse(
	ctx context.Context,
	params core.LLMParams,
) <-chan core.LLMResponse {
	out := make(chan core.LLMResponse)

	go func() {
		defer close(out)

		openaiParams := openai.ChatCompletionNewParams{
			Model:    openai.F(openai.ChatModelGPT4o),
			Messages: openai.F(p.buildMessages(params)),
		}

		if tools := p.buildTools(params, &openaiParams); len(tools) > 0 {
			openaiParams.Tools = openai.F(tools)
		}

		p.buildResponseFormat(params, &openaiParams)

		stream := p.Client.Chat.Completions.NewStreaming(ctx, openaiParams)
		acc := openai.ChatCompletionAccumulator{}

		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			// Check for completed content
			if content, ok := acc.JustFinishedContent(); ok && content != "" {
				out <- core.LLMResponse{
					Content: content,
				}
			}

			// Check for completed tool calls
			if tool, ok := acc.JustFinishedToolCall(); ok {
				// Parse the arguments string as JSON
				var argsMap map[string]interface{}
				if err := json.Unmarshal([]byte(tool.Arguments), &argsMap); err != nil {
					// If parsing fails, use the raw string
					argsMap = map[string]interface{}{
						"raw_args": tool.Arguments,
					}
				}

				out <- core.LLMResponse{
					ToolCalls: []core.ToolCall{
						{
							Name: tool.Name,
							Args: argsMap,
						},
					},
				}
			}

			// Send delta content (if any)
			for _, choice := range chunk.Choices {
				if choice.Delta.Content != "" {
					out <- core.LLMResponse{
						Content: choice.Delta.Content,
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			out <- core.LLMResponse{
				Error: err,
			}
		}
	}()

	return out
}

func (p *OpenAIProvider) buildMessages(
	params core.LLMParams,
) []openai.ChatCompletionMessageParamUnion {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(params.Messages))

	for _, message := range params.Messages {
		switch message.Role {
		case "system":
			messages = append(messages, openai.SystemMessage(message.Content))
		case "user":
			messages = append(messages, openai.UserMessage(message.Content))
		case "assistant":
			messages = append(messages, openai.AssistantMessage(message.Content))
		}
	}

	return messages
}

func (p *OpenAIProvider) buildTools(
	params core.LLMParams,
	openaiParams *openai.ChatCompletionNewParams,
) []openai.ChatCompletionToolParam {
	tools := make([]openai.ChatCompletionToolParam, 0, len(params.Tools))

	if len(params.Tools) > 0 {
		for _, tool := range params.Tools {
			// Get the tool's schema directly from the tool
			schema := tool.Schema()

			// Schema is already of type map[string]interface{}, which is compatible with FunctionParameters

			// Create function parameter from tool's schema
			toolParam := openai.ChatCompletionToolParam{
				Type: openai.F(openai.ChatCompletionToolTypeFunction),
				Function: openai.F(openai.FunctionDefinitionParam{
					Name:        openai.String(tool.Name()),
					Description: openai.String(tool.Description()),
					Parameters:  openai.F(openai.FunctionParameters(schema)), // Cast to FunctionParameters
				}),
			}

			tools = append(tools, toolParam)
		}
		openaiParams.Tools = openai.F(tools)
	}

	return tools
}

func (p *OpenAIProvider) buildResponseFormat(
	params core.LLMParams,
	openaiParams *openai.ChatCompletionNewParams,
) {
	if params.ResponseFormatName != "" {
		schemaParam := openai.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:        openai.F(params.ResponseFormatName),
			Description: openai.F(params.ResponseFormatDescription),
			Schema:      openai.F(params.Schema),
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
