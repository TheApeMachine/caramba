package provider

import (
	"context"
	"encoding/json"
	"os"

	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/spf13/viper"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
DeepseekProvider implements an LLM provider that connects to Deepseek's API.
It supports regular chat completions and streaming responses.
*/
type DeepseekProvider struct {
	client *deepseek.Client
	buffer *stream.Buffer
	params *aiCtx.Artifact
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewDeepseekProvider creates a new Deepseek provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the DEEPSEEK_API_KEY environment variable.
*/
func NewDeepseekProvider(
	apiKey string,
	endpoint string,
) *DeepseekProvider {
	errnie.Debug("provider.NewDeepseekProvider")

	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}

	if endpoint == "" {
		endpoint = viper.GetViper().GetString("endpoints.deepseek")
	}

	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &DeepseekProvider{
		client: deepseek.NewClient(apiKey),
		params: aiCtx.New(
			deepseek.DeepSeekChat,
			nil,
			nil,
			nil,
			0.7,
			1.0,
			0,
			0.0,
			0.0,
			2048,
			false,
		),
		ctx:    ctx,
		cancel: cancel,
	}

	prvdr.buffer = stream.NewBuffer(
		func(event *event.Artifact) error {
			errnie.Debug("provider.DeepseekProvider.buffer.fn", "event", event)

			payload, err := event.Payload()
			if errnie.Error(err) != nil {
				return err
			}

			_, err = prvdr.params.Write(payload)
			if errnie.Error(err) != nil {
				return err
			}

			return nil
		},
	)

	return prvdr
}

/*
Read implements the io.Reader interface.
*/
func (prvdr *DeepseekProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.DeepseekProvider.Read")
	return prvdr.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (prvdr *DeepseekProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.DeepseekProvider.Write")

	n, err = prvdr.buffer.Write(p)
	if errnie.Error(err) != nil {
		return n, err
	}

	composed := &deepseek.ChatCompletionRequest{
		Model: deepseek.DeepSeekChat,
	}

	prvdr.buildMessages(prvdr.params, composed)
	prvdr.buildTools(prvdr.params, composed)
	prvdr.buildResponseFormat(prvdr.params, composed)

	if prvdr.params.Stream() {
		prvdr.handleStreamingRequest(composed)
	} else {
		prvdr.handleSingleRequest(composed)
	}

	return n, nil
}

/*
Close cleans up any resources.
*/
func (prvdr *DeepseekProvider) Close() error {
	errnie.Debug("provider.DeepseekProvider.Close")
	prvdr.cancel()
	return prvdr.params.Close()
}

func (prvdr *DeepseekProvider) buildMessages(
	params *aiCtx.Artifact,
	chatParams *deepseek.ChatCompletionRequest,
) {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "deepseek")
		return
	}

	messages, err := params.Messages()
	if err != nil {
		errnie.Error("failed to get messages", "error", err)
		return
	}

	messageList := make([]deepseek.ChatCompletionMessage, 0, messages.Len())

	for idx := range messages.Len() {
		message := messages.At(idx)

		role, err := message.Role()
		if err != nil {
			errnie.Error("failed to get message role", "error", err)
			continue
		}

		content, err := message.Content()
		if err != nil {
			errnie.Error("failed to get message content", "error", err)
			continue
		}

		switch role {
		case "system":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleSystem,
				Content: content,
			})
		case "user":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleUser,
				Content: content,
			})
		case "assistant":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleAssistant,
				Content: content,
			})
		default:
			errnie.Error("unknown message role", "role", role)
		}
	}

	chatParams.Messages = messageList
}

func (prvdr *DeepseekProvider) buildTools(
	params *aiCtx.Artifact,
	chatParams *deepseek.ChatCompletionRequest,
) {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "deepseek")
		return
	}

	tools, err := params.Tools()
	if err != nil {
		errnie.Error("failed to get tools", "error", err)
		return
	}

	if tools.Len() == 0 {
		return
	}

	toolList := make([]deepseek.Tool, 0, tools.Len())

	for idx := range tools.Len() {
		tool := tools.At(idx)

		name, err := tool.Name()
		if err != nil {
			errnie.Error("failed to get tool name", "error", err)
			continue
		}

		description, err := tool.Description()
		if err != nil {
			errnie.Error("failed to get tool description", "error", err)
			continue
		}

		parameters, err := tool.Parameters()
		if err != nil {
			errnie.Error("failed to get tool parameters", "error", err)
			continue
		}

		properties := make(map[string]interface{})
		for idx := range parameters.Len() {
			param := parameters.At(idx)
			typ, err := param.Type()
			if err != nil {
				continue
			}
			props, err := param.Properties()
			if err != nil {
				continue
			}

			properties[typ] = map[string]interface{}{
				"type":       typ,
				"required":   param.Required(),
				"properties": props,
			}
		}

		required := make([]string, 0)
		for idx := range parameters.Len() {
			param := parameters.At(idx)
			typ, _ := param.Type()
			required = append(required, typ)
		}

		toolList = append(toolList, deepseek.Tool{
			Type: "function",
			Function: deepseek.Function{
				Name:        name,
				Description: description,
				Parameters: &deepseek.FunctionParameters{
					Type:       "object",
					Properties: properties,
					Required:   required,
				},
			},
		})
	}

	if len(toolList) > 0 {
		chatParams.Tools = toolList
	}
}

func (prvdr *DeepseekProvider) buildResponseFormat(
	params *aiCtx.Artifact,
	chatParams *deepseek.ChatCompletionRequest,
) {
	errnie.Debug("provider.buildResponseFormat")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "deepseek")
		return
	}

	data, err := params.Process()
	if err != nil || data == nil {
		return
	}

	var formatData map[string]interface{}
	if err := json.Unmarshal(data, &formatData); err != nil {
		errnie.Error("failed to unmarshal process data", "error", err)
		return
	}

	// Add format instructions as a system message since Deepseek doesn't support direct format control
	if name, ok := formatData["name"].(string); ok {
		if desc, ok := formatData["description"].(string); ok {
			formatMsg := deepseek.ChatCompletionMessage{
				Role: deepseek.ChatMessageRoleSystem,
				Content: "Please format your response according to the specified schema: " +
					name + ". " + desc,
			}
			chatParams.Messages = append(chatParams.Messages, formatMsg)
		}
	}
}

func (prvdr *DeepseekProvider) handleSingleRequest(
	params *deepseek.ChatCompletionRequest,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	response, err := prvdr.client.CreateChatCompletion(prvdr.ctx, params)
	if errnie.Error(err) != nil {
		return
	}

	if len(response.Choices) == 0 {
		errnie.Error("no response choices")
		return
	}

	return utils.SendEvent(
		prvdr.buffer,
		"provider.deepseek",
		message.AssistantRole,
		response.Choices[0].Message.Content,
	)
}

func (prvdr *DeepseekProvider) handleStreamingRequest(
	params *deepseek.ChatCompletionRequest,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	// Convert to stream request
	streamRequest := &deepseek.StreamChatCompletionRequest{
		Model:       params.Model,
		Messages:    params.Messages,
		Temperature: params.Temperature,
		TopP:        params.TopP,
		MaxTokens:   params.MaxTokens,
		Stop:        params.Stop,
		Stream:      true,
	}

	stream, err := prvdr.client.CreateChatCompletionStream(prvdr.ctx, streamRequest)
	if errnie.Error(err) != nil {
		return
	}
	defer stream.Close()

	for {
		response, err := stream.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			errnie.Error("streaming error", "error", err)
			return err
		}

		if len(response.Choices) > 0 {
			content := response.Choices[0].Delta.Content
			if content != "" {
				if err = utils.SendEvent(
					prvdr.buffer,
					"provider.deepseek",
					message.AssistantRole,
					content,
				); errnie.Error(err) != nil {
					continue
				}
			}
		}
	}

	return nil
}

type DeepseekEmbedder struct {
	client *deepseek.Client
	params *aiCtx.Artifact
	ctx    context.Context
}

func NewDeepseekEmbedder(apiKey string) (*DeepseekEmbedder, error) {
	errnie.Debug("provider.NewDeepseekEmbedder")

	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}

	client := deepseek.NewClient(apiKey)

	return &DeepseekEmbedder{
		client: client,
		params: &aiCtx.Artifact{},
		ctx:    context.Background(),
	}, nil
}

func (embedder *DeepseekEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.DeepseekEmbedder.Read", "p", string(p))
	return 0, nil
}

func (embedder *DeepseekEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.DeepseekEmbedder.Write")
	errnie.Warn("Deepseek embedder is not implemented")
	return len(p), nil
}

func (embedder *DeepseekEmbedder) Close() error {
	errnie.Debug("provider.DeepseekEmbedder.Close")
	embedder.params = nil
	return nil
}
