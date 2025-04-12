package provider

import (
	"context"
	"encoding/json"
	"os"

	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
DeepseekProvider implements an LLM provider that connects to Deepseek's API.
It supports regular chat completions and streaming responses.
*/
type DeepseekProvider struct {
	client *deepseek.Client
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewDeepseekProvider creates a new Deepseek provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the DEEPSEEK_API_KEY environment variable.
*/
func NewDeepseekProvider(opts ...DeepseekProviderOption) *DeepseekProvider {
	errnie.Debug("provider.NewDeepseekProvider")

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &DeepseekProvider{
		client: deepseek.NewClient(apiKey),
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *DeepseekProvider) ID() string {
	return "deepseek"
}

type DeepseekProviderOption func(*DeepseekProvider)

func WithDeepseekAPIKey(apiKey string) DeepseekProviderOption {
	return func(prvdr *DeepseekProvider) {
		prvdr.client = deepseek.NewClient(apiKey)
	}
}

func (prvdr *DeepseekProvider) Generate(
	params ProviderParams,
) (ProviderEvent, error) {
	errnie.Info("provider.Generate", "supplier", "deepseek")

	composed := &deepseek.StreamChatCompletionRequest{
		Model:            params.Model,
		Temperature:      float32(params.Temperature),
		TopP:             float32(params.TopP),
		PresencePenalty:  float32(params.PresencePenalty),
		FrequencyPenalty: float32(params.FrequencyPenalty),
		MaxTokens:        int(params.MaxTokens),
	}

	var err error

	if err = prvdr.buildMessages(composed, params.Messages); err != nil {
		return ProviderEvent{}, err
	}

	// Get tools from the artifact metadata
	if err = prvdr.buildTools(composed, params.Tools); err != nil {
		return ProviderEvent{}, err
	}

	if params.ResponseFormat != (ResponseFormat{}) {
		if err = prvdr.buildResponseFormat(composed, params.ResponseFormat); err != nil {
			return ProviderEvent{}, err
		}
	}

	if params.Stream {
		return prvdr.handleStreamingRequest(composed)
	}

	return prvdr.handleSingleRequest(composed)
}

func (prvdr *DeepseekProvider) Name() string {
	return "deepseek"
}

func (prvdr *DeepseekProvider) handleSingleRequest(
	params *deepseek.StreamChatCompletionRequest,
) (ProviderEvent, error) {
	errnie.Debug("provider.handleSingleRequest")

	prms := &deepseek.ChatCompletionRequest{
		Model:       params.Model,
		Messages:    params.Messages,
		Temperature: params.Temperature,
		TopP:        params.TopP,
		MaxTokens:   params.MaxTokens,
	}

	response, err := prvdr.client.CreateChatCompletion(prvdr.ctx, prms)
	if errnie.Error(err) != nil {
		return ProviderEvent{}, err
	}

	if len(response.Choices) == 0 {
		err = errnie.Error("no response choices")
		return ProviderEvent{}, err
	}

	// Return provider event with message
	return ProviderEvent{
		Message: Message{
			Role:    "assistant",
			Name:    string(params.Model),
			Content: response.Choices[0].Message.Content,
		},
	}, nil
}

func (prvdr *DeepseekProvider) handleStreamingRequest(
	params *deepseek.StreamChatCompletionRequest,
) (ProviderEvent, error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream, err := prvdr.client.CreateChatCompletionStream(prvdr.ctx, params)
	if errnie.Error(err) != nil {
		return ProviderEvent{}, err
	}
	defer stream.Close()

	var lastContent string
	for {
		response, err := stream.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			errnie.Error("streaming error", "error", err)
			return ProviderEvent{}, err
		}

		if len(response.Choices) > 0 {
			content := response.Choices[0].Delta.Content
			if content != "" {
				lastContent = content
			}
		}
	}

	// Return the last chunk
	return ProviderEvent{
		Message: Message{
			Role:    "assistant",
			Name:    string(params.Model),
			Content: lastContent,
		},
	}, nil
}

func (prvdr *DeepseekProvider) buildMessages(
	chatParams *deepseek.StreamChatCompletionRequest,
	messages []Message,
) (err error) {
	errnie.Debug("provider.buildMessages")

	messageList := make([]deepseek.ChatCompletionMessage, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleSystem,
				Content: msg.Content,
			})
		case "user":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleUser,
				Content: msg.Content,
			})
		case "assistant":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleAssistant,
				Content: msg.Content,
			})
		default:
			errnie.Error("unknown message role", "role", msg.Role)
		}
	}

	chatParams.Messages = messageList

	return nil
}

func (prvdr *DeepseekProvider) buildTools(
	chatParams *deepseek.StreamChatCompletionRequest,
	tools []mcp.Tool,
) (err error) {
	errnie.Debug("provider.buildTools")

	if len(tools) == 0 {
		return nil
	}

	toolList := make([]deepseek.Tool, 0, len(tools))

	for _, tool := range tools {
		properties := make(map[string]any)

		for name, prop := range tool.InputSchema.Properties {
			properties[name] = prop
		}

		toolList = append(toolList, deepseek.Tool{
			Type: "function",
			Function: deepseek.Function{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters: &deepseek.FunctionParameters{
					Type:       "object",
					Properties: properties,
					Required:   tool.InputSchema.Required,
				},
			},
		})
	}

	if len(toolList) > 0 {
		chatParams.Tools = toolList
	}

	return nil
}

func (prvdr *DeepseekProvider) buildResponseFormat(
	chatParams *deepseek.StreamChatCompletionRequest,
	format ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	// If no format specified, skip
	if format.Name == "" && format.Description == "" && format.Schema == nil {
		return nil
	}

	schemaStr := ""
	if format.Schema != nil {
		switch s := format.Schema.(type) {
		case string:
			schemaStr = s
		default:
			j, _ := json.Marshal(format.Schema)
			schemaStr = string(j)
		}
	}

	// Add format instructions as a system message since Deepseek doesn't support direct format control
	formatMsg := deepseek.ChatCompletionMessage{
		Role: deepseek.ChatMessageRoleSystem,
		Content: "Please format your response according to the following schema: " +
			schemaStr + ". " + format.Description,
	}
	chatParams.Messages = append(chatParams.Messages, formatMsg)

	return nil
}
