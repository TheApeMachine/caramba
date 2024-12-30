package provider

import (
	"context"
	"errors"
	"fmt"

	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/theapemachine/errnie"
)

type OpenAI struct {
	client *sdk.Client
	model  string
}

func NewOpenAI(apiKey string) *OpenAI {
	return &OpenAI{
		client: sdk.NewClient(),
		model:  sdk.ChatModelGPT4o,
	}
}

func NewOpenAICompatible(apiKey, endpoint, model string) *OpenAI {
	return &OpenAI{
		client: sdk.NewClient(option.WithAPIKey(apiKey), option.WithBaseURL(endpoint)),
		model:  model,
	}
}

func (openai *OpenAI) Name() string {
	return fmt.Sprintf("openai (%s)", openai.model)
}

func (openai *OpenAI) Generate(ctx context.Context, params *GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		tools := openai.convertTools(params)
		messages := openai.convertMessages(params)
		processes := openai.convertProcesses(params)

		// Only proceed if we have messages
		if len(messages) == 0 {
			errnie.Error(errors.New("no valid messages to process"))
			out <- Event{Type: EventError, Error: errors.New("no valid messages to process")}
			return
		}

		// Build request with required fields
		request := sdk.ChatCompletionNewParams{
			Model:    sdk.F(openai.model),
			Messages: sdk.F(messages),
		}

		// Add optional parameters
		if params.MaxTokens > 0 {
			request.MaxTokens = sdk.F(params.MaxTokens)
		}
		if params.Temperature > 0 {
			request.Temperature = sdk.F(params.Temperature)
		}
		if len(tools) > 0 {
			request.Tools = sdk.F(tools)
		}
		if processes != nil {
			request.ResponseFormat = sdk.F[sdk.ChatCompletionNewParamsResponseFormatUnion](processes)
		}

		stream := openai.client.Chat.Completions.NewStreaming(ctx, request)
		acc := sdk.ChatCompletionAccumulator{}

		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			// Handle content streaming
			if content, ok := acc.JustFinishedContent(); ok {
				out <- Event{Type: EventDone, Text: content}
			}

			// Handle tool calls
			if tool, ok := acc.JustFinishedToolCall(); ok {
				out <- Event{
					Type:        EventChunk,
					PartialJSON: tool.Arguments,
				}
			}

			// Stream regular content chunks
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				out <- Event{
					Type:        EventChunk,
					Text:        chunk.Choices[0].Delta.Content,
					PartialJSON: chunk.Choices[0].Delta.JSON.RawJSON(),
				}
			}
		}

		if err := stream.Err(); err != nil {
			errnie.Error(err)
			out <- Event{Type: EventError, Error: err}
			return
		}

		out <- Event{Type: EventDone, Text: "\n"}
	}()

	return out
}

func (*OpenAI) convertProcesses(params *GenerationParams) *sdk.ResponseFormatJSONSchemaParam {
	if params.Process == nil {
		return nil
	}

	return &sdk.ResponseFormatJSONSchemaParam{
		Type: sdk.F(sdk.ResponseFormatJSONSchemaTypeJSONSchema),
		JSONSchema: sdk.F(sdk.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:        sdk.F(params.Process.Name()),
			Description: sdk.F(params.Process.Description()),
			Schema:      sdk.F(params.Process.GenerateSchema()),
			Strict:      sdk.Bool(true),
		}),
	}
}

func (*OpenAI) convertMessages(params *GenerationParams) []sdk.ChatCompletionMessageParamUnion {
	var messages []sdk.ChatCompletionMessageParamUnion
	if len(params.Thread.Messages) > 0 {
		messages = make([]sdk.ChatCompletionMessageParamUnion, 0, len(params.Thread.Messages))
		for _, msg := range params.Thread.Messages {
			if msg.Content == "" {
				continue
			}

			switch msg.Role {
			case RoleSystem:
				messages = append(messages, sdk.SystemMessage(msg.Content))
			case RoleUser:
				messages = append(messages, sdk.UserMessage(msg.Content))
			case RoleAssistant:
				messages = append(messages, sdk.AssistantMessage(msg.Content))
			}
		}
	}
	return messages
}

func (*OpenAI) convertTools(params *GenerationParams) []sdk.ChatCompletionToolParam {
	var tools []sdk.ChatCompletionToolParam
	if len(params.Tools) > 0 {
		tools = make([]sdk.ChatCompletionToolParam, len(params.Tools))
		for i, tool := range params.Tools {
			tools[i] = sdk.ChatCompletionToolParam{
				Type: sdk.F(sdk.ChatCompletionToolTypeFunction),
				Function: sdk.F(sdk.FunctionDefinitionParam{
					Name:        sdk.F(tool.Name),
					Description: sdk.F(tool.Description),
					Parameters:  sdk.F(tool.Schema.(sdk.FunctionParameters)),
				}),
			}
		}
	}

	return tools
}
