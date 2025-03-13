package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"os"

	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
DeepseekProvider implements an LLM provider that connects to Deepseek's API.
It supports regular chat completions and streaming responses.
*/
type DeepseekProvider struct {
	*ProviderData
	client *deepseek.Client
	buffer *bufio.ReadWriter
	enc    *json.Encoder
	dec    *json.Decoder
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

	var client *deepseek.Client
	if endpoint == "" {
		client = deepseek.NewClient(apiKey)
	} else {
		client = deepseek.NewClient(apiKey, endpoint)
	}

	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	p := &DeepseekProvider{
		ProviderData: &ProviderData{
			Params: &ai.ContextData{},
			Result: &core.EventData{},
		},
		client: client,
		buffer: buffer,
		enc:    json.NewEncoder(buffer),
		dec:    json.NewDecoder(buffer),
	}

	return p
}

/*
Read implements the io.Reader interface.
*/
func (provider *DeepseekProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.DeepseekProvider.Read")

	if err = provider.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = provider.buffer.Read(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("provider.DeepseekProvider.Read", "n", n, "err", err)
	return n, err
}

/*
Write implements the io.Writer interface.
*/
func (provider *DeepseekProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.DeepseekProvider.Write", "p", string(p))

	if n, err = provider.buffer.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = json.Unmarshal(p, provider.ProviderData.Params); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	errnie.Debug("provider.DeepseekProvider.Write", "n", n, "err", err)

	// Create the Deepseek request
	deepseekParams := &deepseek.ChatCompletionRequest{
		Model:    deepseek.DeepSeekChat,
		Messages: provider.buildMessages(provider.ProviderData.Params),
	}

	errnie.Debug("provider.DeepseekProvider.Write", "deepseekParams", deepseekParams)

	provider.buildSettings(provider.ProviderData.Params, deepseekParams)

	if provider.ProviderData.Params.Stream {
		err = errnie.NewErrIO(provider.handleStreamingRequest(deepseekParams))
	} else {
		err = errnie.NewErrIO(provider.handleNonStreamingRequest(deepseekParams))
	}

	return n, err
}

/*
Close cleans up any resources.
*/
func (provider *DeepseekProvider) Close() error {
	errnie.Debug("provider.DeepseekProvider.Close")

	// Reset state
	provider.ProviderData.Params = nil
	provider.ProviderData.Result = nil

	provider.buffer = nil
	provider.enc = nil
	provider.dec = nil

	return nil
}

func (p *DeepseekProvider) buildMessages(
	params *ai.ContextData,
) []deepseek.ChatCompletionMessage {
	errnie.Debug("provider.DeepseekProvider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "deepseek")
		return nil
	}

	messages := make([]deepseek.ChatCompletionMessage, 0, len(params.Messages))

	for _, message := range params.Messages {
		var role string
		switch message.Role {
		case "system":
			role = deepseek.ChatMessageRoleSystem
		case "user":
			role = deepseek.ChatMessageRoleUser
		case "assistant":
			role = deepseek.ChatMessageRoleAssistant
		default:
			errnie.Error("unknown message role", "role", message.Role)
			continue
		}

		messages = append(messages, deepseek.ChatCompletionMessage{
			Role:    role,
			Content: message.Content,
		})
	}

	return messages
}

func (p *DeepseekProvider) buildSettings(
	params *ai.ContextData,
	deepseekParams *deepseek.ChatCompletionRequest,
) {
	errnie.Debug("provider.DeepseekProvider.buildSettings")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "deepseek")
		return
	}

	// Set temperature if provided (float32 conversion)
	if params.Temperature > 0 {
		deepseekParams.Temperature = float32(params.Temperature)
	}

	// Set max tokens if provided
	if params.MaxTokens > 0 {
		deepseekParams.MaxTokens = params.MaxTokens
	}

	// Set top_p if provided (float32 conversion)
	if params.TopP > 0 {
		deepseekParams.TopP = float32(params.TopP)
	}

	// Set stop sequences if provided
	if len(params.StopSequences) > 0 {
		deepseekParams.Stop = params.StopSequences
	}

	// Set JSON mode if structured output is requested
	if params.Process != nil && params.Process.ProcessData != nil && params.Process.ProcessData.Schema != nil {
		deepseekParams.JSONMode = true
	}
}

func (p *DeepseekProvider) handleNonStreamingRequest(
	deepseekParams *deepseek.ChatCompletionRequest,
) error {
	errnie.Debug("provider.DeepseekProvider.handleNonStreamingRequest")

	ctx := context.Background()
	response, err := p.client.CreateChatCompletion(ctx, deepseekParams)
	if err != nil {
		errnie.Error("failed to get response from Deepseek", "error", err)
		return err
	}

	// Extract the response content
	if len(response.Choices) > 0 {
		assistantMessage := response.Choices[0].Message.Content
		message := core.NewMessage("assistant", "", assistantMessage)
		p.ProviderData.Result.Message = message.MessageData
	}

	// Handle token usage if needed (seems core.Event doesn't have a Tokens field)
	// Usage information is available in response.Usage if needed

	// Encode the result back to the buffer
	if err := p.enc.Encode(p.ProviderData.Result); err != nil {
		errnie.Error("failed to encode result", "error", err)
		return err
	}

	return nil
}

func (p *DeepseekProvider) handleStreamingRequest(
	deepseekParams *deepseek.ChatCompletionRequest,
) error {
	errnie.Debug("provider.DeepseekProvider.handleStreamingRequest")

	// Convert to stream request
	streamRequest := &deepseek.StreamChatCompletionRequest{
		Model:       deepseekParams.Model,
		Messages:    deepseekParams.Messages,
		Temperature: deepseekParams.Temperature,
		TopP:        deepseekParams.TopP,
		MaxTokens:   deepseekParams.MaxTokens,
		Stop:        deepseekParams.Stop,
		Stream:      true,
	}

	ctx := context.Background()
	stream, err := p.client.CreateChatCompletionStream(ctx, streamRequest)
	if err != nil {
		errnie.Error("failed to get stream from Deepseek", "error", err)
		return err
	}
	defer stream.Close()

	var fullMessage string
	for {
		response, err := stream.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			errnie.Error("failed to receive from stream", "error", err)
			return err
		}

		if len(response.Choices) > 0 {
			content := response.Choices[0].Delta.Content
			fullMessage += content

			// Create a partial result for the streaming chunk
			partialEvent := core.NewEvent(
				core.NewMessage("assistant", "", content),
				nil,
			).EventData

			// Encode the partial result back to the buffer
			if err := p.enc.Encode(partialEvent); err != nil {
				errnie.Error("failed to encode partial result", "error", err)
				return err
			}

			// Flush the buffer to make it available for reading
			if err := p.buffer.Flush(); err != nil {
				errnie.Error("failed to flush buffer", "error", err)
				return err
			}
		}
	}

	// Create the final result
	finalEvent := core.NewEvent(
		core.NewMessage("assistant", "", fullMessage),
		nil,
	).EventData

	// Encode the final result back to the buffer
	if err := p.enc.Encode(finalEvent); err != nil {
		errnie.Error("failed to encode final result", "error", err)
		return err
	}

	return nil
}
