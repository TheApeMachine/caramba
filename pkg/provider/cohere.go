package provider

import (
	"context"
	"encoding/json"
	"os"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/spf13/viper"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
CohereProvider implements an LLM provider that connects to Cohere's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type CohereProvider struct {
	client *cohereclient.Client
	buffer *stream.Buffer
	params *aiCtx.Artifact
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewCohereProvider creates a new Cohere provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the COHERE_API_KEY environment variable.
*/
func NewCohereProvider(
	apiKey string,
	endpoint string,
) *CohereProvider {
	errnie.Debug("provider.NewCohereProvider")

	if apiKey == "" {
		apiKey = os.Getenv("COHERE_API_KEY")
	}

	if endpoint == "" {
		endpoint = viper.GetViper().GetString("endpoints.cohere")
	}

	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &CohereProvider{
		client: cohereclient.NewClient(
			cohereclient.WithToken(apiKey),
		),
		params: aiCtx.New(
			"command", // Default model
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
			errnie.Debug("provider.CohereProvider.buffer.fn", "event", event)

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
func (prvdr *CohereProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereProvider.Read")
	return prvdr.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (prvdr *CohereProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereProvider.Write")

	n, err = prvdr.buffer.Write(p)
	if errnie.Error(err) != nil {
		return n, err
	}

	composed := &cohere.ChatStreamRequest{}

	model, err := prvdr.params.Model()
	if err != nil {
		errnie.Error("failed to get model", "error", err)
		return n, err
	}

	composed.Model = cohere.String(model)

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
func (prvdr *CohereProvider) Close() error {
	errnie.Debug("provider.CohereProvider.Close")
	prvdr.cancel()
	return prvdr.params.Close()
}

func (prvdr *CohereProvider) buildMessages(
	params *aiCtx.Artifact,
	chatParams *cohere.ChatStreamRequest,
) {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "cohere")
		return
	}

	messages, err := params.Messages()
	if err != nil {
		errnie.Error("failed to get messages", "error", err)
		return
	}

	var systemPrompt string
	messageList := make([]*cohere.Message, 0, messages.Len())

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
			systemPrompt = content
		case "user":
			messageList = append(messageList, &cohere.Message{
				Role: "user",
				User: &cohere.ChatMessage{
					Message: content,
				},
			})
		case "assistant":
			messageList = append(messageList, &cohere.Message{
				Role: "chatbot",
				Chatbot: &cohere.ChatMessage{
					Message: content,
				},
			})
		default:
			errnie.Error("unknown message role", "role", role)
		}
	}

	if systemPrompt != "" {
		chatParams.Preamble = cohere.String(systemPrompt)
	}
	chatParams.ChatHistory = messageList
}

func (prvdr *CohereProvider) buildTools(
	params *aiCtx.Artifact,
	chatParams *cohere.ChatStreamRequest,
) {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "cohere")
		return
	}

	tools, err := params.Tools()
	if err != nil {
		errnie.Error("failed to get tools", "error", err)
		return
	}

	toolList := make([]*cohere.Tool, 0, tools.Len())

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

		toolList = append(toolList, &cohere.Tool{
			Name:        name,
			Description: description,
		})
	}

	if len(toolList) > 0 {
		chatParams.Tools = toolList
	}
}

func (prvdr *CohereProvider) buildResponseFormat(
	params *aiCtx.Artifact,
	chatParams *cohere.ChatStreamRequest,
) {
	errnie.Debug("provider.buildResponseFormat")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "cohere")
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

	// Note: Cohere doesn't support response format directly like OpenAI
	// We'll add it to the system prompt if needed
	if name, ok := formatData["name"].(string); ok {
		if desc, ok := formatData["description"].(string); ok {
			existingPreamble := ""
			if chatParams.Preamble != nil {
				existingPreamble = *chatParams.Preamble + "\n\n"
			}

			formatInstructions := existingPreamble +
				"Please format your response according to the specified schema: " +
				name + ". " + desc

			chatParams.Preamble = cohere.String(formatInstructions)
		}
	}
}

func (prvdr *CohereProvider) handleSingleRequest(
	params *cohere.ChatStreamRequest,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	// Convert stream request to regular chat request
	chatRequest := &cohere.ChatRequest{
		Model:       params.Model,
		Message:     params.Message,
		ChatHistory: params.ChatHistory,
		Preamble:    params.Preamble,
		Tools:       params.Tools,
		Temperature: params.Temperature,
	}

	response, err := prvdr.client.Chat(prvdr.ctx, chatRequest)
	if errnie.Error(err) != nil {
		return
	}

	return utils.SendEvent(
		prvdr.buffer,
		"provider.cohere",
		message.AssistantRole,
		response.Text,
	)
}

func (prvdr *CohereProvider) handleStreamingRequest(
	params *cohere.ChatStreamRequest,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream, err := prvdr.client.ChatStream(prvdr.ctx, params)
	if errnie.Error(err) != nil {
		return
	}
	defer stream.Close()

	for {
		event, err := stream.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			errnie.Error("streaming error", "error", err)
			return err
		}

		if content := event.TextGeneration.String(); content != "" {
			if err = utils.SendEvent(
				prvdr.buffer,
				"provider.cohere",
				message.AssistantRole,
				content,
			); errnie.Error(err) != nil {
				continue
			}
		}
	}

	return nil
}

type CohereEmbedder struct {
	params   *aiCtx.Artifact
	apiKey   string
	endpoint string
	client   *cohereclient.Client
}

func NewCohereEmbedder(apiKey string, endpoint string) *CohereEmbedder {
	errnie.Debug("provider.NewCohereEmbedder")

	if apiKey == "" {
		apiKey = os.Getenv("COHERE_API_KEY")
	}

	return &CohereEmbedder{
		params:   &aiCtx.Artifact{},
		apiKey:   apiKey,
		endpoint: endpoint,
		client:   cohereclient.NewClient(cohereclient.WithToken(apiKey)),
	}
}

func (embedder *CohereEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereEmbedder.Read", "p", string(p))
	return 0, nil
}

func (embedder *CohereEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereEmbedder.Write")

	messages, err := embedder.params.Messages()
	if err != nil {
		errnie.Error("failed to get messages", "error", err)
		return 0, err
	}

	if messages.Len() == 0 {
		return len(p), nil
	}

	message := messages.At(0)
	content, err := message.Content()
	if err != nil {
		errnie.Error("failed to get message content", "error", err)
		return 0, err
	}

	in := cohere.EmbedInputType(cohere.EmbedInputTypeSearchDocument)

	embedRequest := cohere.EmbedRequest{
		Texts:     []string{content},
		Model:     cohere.String("embed-english-v3.0"),
		InputType: &in,
	}

	response, err := embedder.client.Embed(context.Background(), &embedRequest)
	if err != nil {
		errnie.Error("embedding request failed", "error", err)
		return 0, err
	}

	if response != nil && len(response.EmbeddingsFloats.Embeddings) > 0 {
		embeddings := response.EmbeddingsFloats.Embeddings
		errnie.Debug("created embeddings",
			"text_length", len(content),
			"dimensions", len(embeddings),
		)
	}

	return len(p), nil
}

func (embedder *CohereEmbedder) Close() error {
	errnie.Debug("provider.CohereEmbedder.Close")
	embedder.params = nil
	return nil
}
