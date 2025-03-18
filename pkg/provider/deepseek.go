package provider

import (
	"context"
	"os"

	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/spf13/viper"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
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
	params *Params
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
	params := &Params{
		Model:       deepseek.DeepSeekChat,
		Temperature: 0.7,
		TopP:        1.0,
		MaxTokens:   2048,
	}

	prvdr := &DeepseekProvider{
		client: deepseek.NewClient(apiKey),
		buffer: stream.NewBuffer(func(artfct *datura.Artifact) (err error) {
			var payload []byte

			if payload, err = artfct.EncryptedPayload(); err != nil {
				return errnie.Error(err)
			}

			params.Unmarshal(payload)
			return nil
		}),
		params: params,
		ctx:    ctx,
		cancel: cancel,
	}

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

	prvdr.buildMessages(composed)
	prvdr.buildTools(composed)
	prvdr.buildResponseFormat(composed)

	if prvdr.params.Stream {
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
	return nil
}

func (prvdr *DeepseekProvider) buildMessages(
	chatParams *deepseek.ChatCompletionRequest,
) {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "deepseek")
		return
	}

	messageList := make([]deepseek.ChatCompletionMessage, 0, len(prvdr.params.Messages))

	for _, message := range prvdr.params.Messages {
		switch message.Role {
		case "system":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleSystem,
				Content: message.Content,
			})
		case "user":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleUser,
				Content: message.Content,
			})
		case "assistant":
			messageList = append(messageList, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleAssistant,
				Content: message.Content,
			})
		default:
			errnie.Error("unknown message role", "role", message.Role)
		}
	}

	chatParams.Messages = messageList
}

func (prvdr *DeepseekProvider) buildTools(
	chatParams *deepseek.ChatCompletionRequest,
) {
	errnie.Debug("provider.buildTools")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "deepseek")
		return
	}

	if len(prvdr.params.Tools) == 0 {
		return
	}

	toolList := make([]deepseek.Tool, 0, len(prvdr.params.Tools))

	for _, tool := range prvdr.params.Tools {
		properties := make(map[string]interface{})

		for _, prop := range tool.Function.Parameters.Properties {
			properties[prop.Name] = map[string]interface{}{
				"type":        prop.Type,
				"description": prop.Description,
			}

			if len(prop.Enum) > 0 {
				properties[prop.Name].(map[string]interface{})["enum"] = prop.Enum
			}
		}

		toolList = append(toolList, deepseek.Tool{
			Type: "function",
			Function: deepseek.Function{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters: &deepseek.FunctionParameters{
					Type:       "object",
					Properties: properties,
					Required:   tool.Function.Parameters.Required,
				},
			},
		})
	}

	if len(toolList) > 0 {
		chatParams.Tools = toolList
	}
}

func (prvdr *DeepseekProvider) buildResponseFormat(
	chatParams *deepseek.ChatCompletionRequest,
) {
	errnie.Debug("provider.buildResponseFormat")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "deepseek")
		return
	}

	// Add format instructions as a system message since Deepseek doesn't support direct format control
	if prvdr.params.ResponseFormat.Name != "" {
		formatMsg := deepseek.ChatCompletionMessage{
			Role: deepseek.ChatMessageRoleSystem,
			Content: "Please format your response according to the specified schema: " +
				prvdr.params.ResponseFormat.Name + ". " + prvdr.params.ResponseFormat.Description,
		}
		chatParams.Messages = append(chatParams.Messages, formatMsg)
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
