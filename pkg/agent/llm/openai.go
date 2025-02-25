package llm

import (
	"context"

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
) (string, error) {
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
		return "", err
	}

	return completion.Choices[0].Message.Content, nil
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

			// When this fires, the current chunk value will not contain content data
			if content, ok := acc.JustFinishedContent(); ok {
				out <- core.LLMResponse{
					Content: content,
				}
			}

			if tool, ok := acc.JustFinishedToolCall(); ok {
				out <- core.LLMResponse{
					ToolCalls: []core.ToolCall{
						{
							Name: tool.Name,
							Args: map[string]interface{}{
								"args": tool.Arguments,
							},
						},
					},
				}
			}

			if refusal, ok := acc.JustFinishedRefusal(); ok {
				out <- core.LLMResponse{
					Refusal: refusal,
				}
			}

			// It's best to use chunks after handling JustFinished events
			if len(chunk.Choices) > 0 {
				out <- core.LLMResponse{
					Content: chunk.Choices[0].Delta.JSON.RawJSON(),
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
			tools = append(tools, openai.ChatCompletionToolParam{
				Type: openai.F(openai.ChatCompletionToolTypeFunction),
				Function: openai.F(openai.FunctionDefinitionParam{
					Name:        openai.String(tool.Name()),
					Description: openai.String(tool.Description()),
					Parameters: openai.F(openai.FunctionParameters{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]string{
								"type": "string",
							},
						},
						"required": []string{"location"},
					}),
				}),
			})
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
