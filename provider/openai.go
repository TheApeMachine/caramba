package provider

import (
	"context"
	"errors"
	"fmt"

	"github.com/charmbracelet/log"
	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/theapemachine/errnie"
)

type OpenAI struct {
	*BaseProvider
	client *sdk.Client
	model  string
	cancel context.CancelFunc
}

func NewOpenAI(apiKey string) *OpenAI {
	return &OpenAI{
		BaseProvider: NewBaseProvider(),
		client:       sdk.NewClient(),
		model:        sdk.ChatModelGPT4oMini,
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

func (openai *OpenAI) Generate(ctx context.Context, params *LLMGenerationParams) <-chan *Event {
	out := make(chan *Event)
	ctx, cancel := context.WithCancel(ctx)
	openai.cancel = cancel

	go func() {
		defer close(out)
		defer cancel()

		// Send start event
		startEvent := NewEvent("openai:gpt-4o-mini", EventStart, "", "", nil)
		out <- startEvent

		tools := openai.convertTools(params)
		messages := openai.convertMessages(params)
		processes := openai.convertProcesses(params)

		// Only proceed if we have messages
		if len(messages) == 0 {
			err := errors.New("no valid messages to process")
			errnie.Error(err)
			event := NewEvent("openai:gpt-4o-mini", EventError, "", err.Error(), nil)
			out <- event
			return
		}

		// Create completion request
		// Create completion request
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

		// Stream the response
		stream := openai.client.Chat.Completions.NewStreaming(ctx, request)
		acc := sdk.ChatCompletionAccumulator{}

		for stream.Next() {
			select {
			case <-ctx.Done():
				return
			default:
				chunk := stream.Current()
				acc.AddChunk(chunk)

				// Handle finished content from accumulator
				if content, ok := acc.JustFinishedContent(); ok {
					chunkEvent := NewEvent("openai:gpt-4o-mini", EventChunk, content, "", nil)
					out <- chunkEvent
					continue
				}

				// Handle tool calls from accumulator
				if tool, ok := acc.JustFinishedToolCall(); ok {
					toolEvent := NewEvent("openai:gpt-4o-mini", EventFunction, "", tool.Arguments, nil)
					out <- toolEvent
					continue
				}

				// Handle refusals from accumulator
				if refusal, ok := acc.JustFinishedRefusal(); ok {
					refusalEvent := NewEvent("openai:gpt-4o-mini", EventError, "", refusal, nil)
					out <- refusalEvent
					continue
				}

				// Extract content from delta if available
				if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
					chunkEvent := NewEvent(
						"openai:gpt-4o-mini",
						EventChunk,
						chunk.Choices[0].Delta.Content,
						chunk.Choices[0].Delta.JSON.RawJSON(),
						nil,
					)
					out <- chunkEvent
				}
			}
		}

		if err := stream.Err(); err != nil {
			log.Error("Error streaming OpenAI response", "error", err)
			errEvent := NewEvent("openai:gpt-4o-mini", EventError, "", err.Error(), nil)
			out <- errEvent
			return
		}

		// Send done event
		doneEvent := NewEvent("generate:stop", EventStop, "\n", "", nil)
		out <- doneEvent
	}()

	return out
}

func (openai *OpenAI) CancelGeneration(ctx context.Context) error {
	if openai.cancel != nil {
		openai.cancel()
	}
	return nil
}

func (openai *OpenAI) Cleanup(ctx context.Context) error {
	if openai.client != nil {
		// No specific cleanup needed for OpenAI client
		openai.client = nil
	}
	return nil
}

func (*OpenAI) convertProcesses(params *LLMGenerationParams) *sdk.ResponseFormatJSONSchemaParam {
	if params.Process == nil {
		return nil
	}

	return &sdk.ResponseFormatJSONSchemaParam{
		Type: sdk.F(sdk.ResponseFormatJSONSchemaTypeJSONSchema),
		JSONSchema: sdk.F(sdk.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:        sdk.F(params.Process.Name()),
			Description: sdk.F(params.Process.Description()),
			Schema:      sdk.F(params.Process.GenerateSchema()),
			Strict:      sdk.Bool(false),
		}),
	}
}

func (*OpenAI) convertMessages(params *LLMGenerationParams) []sdk.ChatCompletionMessageParamUnion {
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

func (*OpenAI) convertTools(params *LLMGenerationParams) []sdk.ChatCompletionToolParam {
	var tools []sdk.ChatCompletionToolParam
	if len(params.Tools) > 0 {
		tools = make([]sdk.ChatCompletionToolParam, len(params.Tools))
		for i, tool := range params.Tools {
			tools[i] = sdk.ChatCompletionToolParam{
				Type: sdk.F(sdk.ChatCompletionToolTypeFunction),
				Function: sdk.F(sdk.FunctionDefinitionParam{
					Name:        sdk.F(tool.Name()),
					Description: sdk.F(tool.Description()),
					Parameters:  sdk.F(tool.GenerateSchema().(sdk.FunctionParameters)),
				}),
			}
		}
	}

	return tools
}

// ValidateConfig validates the provider configuration
func (openai *OpenAI) ValidateConfig() error {
	return nil
}

// Version returns the provider version
func (openai *OpenAI) Version() string {
	return "1.0.0"
}

// Initialize sets up the provider
func (openai *OpenAI) Initialize(ctx context.Context) error {
	return nil
}

// PauseGeneration pauses the current generation
func (openai *OpenAI) PauseGeneration() error {
	return nil
}

// ResumeGeneration resumes the current generation
func (openai *OpenAI) ResumeGeneration() error {
	return nil
}

// GetCapabilities returns the provider capabilities
func (openai *OpenAI) GetCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"streaming": true,
		"tools":     true,
	}
}

// SupportsFeature checks if a feature is supported
func (openai *OpenAI) SupportsFeature(feature string) bool {
	caps := openai.GetCapabilities()
	supported, ok := caps[feature].(bool)
	return ok && supported
}
