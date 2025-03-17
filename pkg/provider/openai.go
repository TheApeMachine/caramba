package provider

import (
	"context"
	"encoding/json"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/spf13/viper"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	client *openai.Client
	buffer *stream.Buffer
	params *aiCtx.Artifact
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewOpenAIProvider creates a new OpenAI provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIProvider(
	apiKey string,
	endpoint string,
) *OpenAIProvider {
	errnie.Debug("provider.NewOpenAIProvider")

	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	if endpoint == "" {
		endpoint = viper.GetViper().GetString("endpoints.openai")
	}

	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &OpenAIProvider{
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
		),
		params: aiCtx.New(
			openai.ChatModelGPT4oMini,
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
			errnie.Debug("provider.OpenAIProvider.buffer.fn", "event", event)

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
func (prvdr *OpenAIProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIProvider.Read")
	return prvdr.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (prvdr *OpenAIProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIProvider.Write")

	n, err = prvdr.buffer.Write(p)
	if errnie.Error(err) != nil {
		return n, err
	}

	composed := openai.ChatCompletionNewParams{}

	// Set the model first with detailed error logging
	model, err := prvdr.params.Model()
	if err != nil {
		errnie.Error("failed to get model", "error", err, "params", prvdr.params)
		return n, err
	}

	composed.Model = openai.F(model)

	prvdr.buildMessages(prvdr.params, &composed)
	prvdr.buildTools(prvdr.params, &composed)
	prvdr.buildResponseFormat(prvdr.params, &composed)

	if prvdr.params.Stream() {
		prvdr.handleStreamingRequest(&composed)
	} else {
		prvdr.handleSingleRequest(&composed)
	}

	return n, nil
}

/*
Close cleans up any resources.
*/
func (prvdr *OpenAIProvider) Close() error {
	errnie.Debug("provider.OpenAIProvider.Close")
	return prvdr.params.Close()
}

// buildMessages converts ContextData messages to OpenAI API format
func (prvdr *OpenAIProvider) buildMessages(
	params *aiCtx.Artifact,
	composed *openai.ChatCompletionNewParams,
) {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "openai")
		return
	}

	messages, err := params.Messages()
	if err != nil {
		errnie.Error("failed to get messages", "error", err)
		return
	}

	messageList := make([]openai.ChatCompletionMessageParamUnion, 0, messages.Len())

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
			messageList = append(messageList, openai.SystemMessage(content))
		case "user":
			messageList = append(messageList, openai.UserMessage(content))
		case "assistant":
			messageList = append(messageList, openai.AssistantMessage(content))
		default:
			errnie.Error("unknown message role", "role", role)
		}
	}

	composed.Messages = openai.F(messageList)
}

// handleSingleRequest processes a single (non-streaming) completion request
func (prvdr *OpenAIProvider) handleSingleRequest(
	params *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	var completion *openai.ChatCompletion

	if completion, err = prvdr.client.Chat.Completions.New(
		prvdr.ctx, *params,
	); errnie.Error(err) != nil {
		return err
	}

	return utils.SendEvent(
		prvdr.buffer,
		"provider.openai",
		message.AssistantRole,
		completion.Choices[0].Message.Content,
	)
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *OpenAIProvider) handleStreamingRequest(
	params *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream := prvdr.client.Chat.Completions.NewStreaming(prvdr.ctx, *params)
	defer stream.Close()

	acc := openai.ChatCompletionAccumulator{}

	for stream.Next() {
		chunk := stream.Current()

		if ok := acc.AddChunk(chunk); !ok {
			errnie.Error("chunk dropped", "id", acc.ID)
			continue
		}

		// When this fires, the current chunk value will not contain content data
		if content, ok := acc.JustFinishedContent(); ok {
			if err = utils.SendEvent(
				prvdr.buffer,
				"provider.openai",
				message.AssistantRole,
				content,
			); errnie.Error(err) != nil {
				continue
			}
		}

		// Handle delta content
		if chunk.Choices[0].Delta.Content != "" {
			if err = utils.SendEvent(
				prvdr.buffer,
				"provider.openai",
				message.AssistantRole,
				chunk.Choices[0].Delta.Content,
			); errnie.Error(err) != nil {
				continue
			}
		}
	}

	if err = stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		return
	}

	return nil
}

func (prvdr *OpenAIProvider) buildTools(
	params *aiCtx.Artifact,
	openaiParams *openai.ChatCompletionNewParams,
) []openai.ChatCompletionToolParam {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "openai")
		return nil
	}

	toolsOut := make([]openai.ChatCompletionToolParam, 0)

	tools, err := params.Tools()

	if err != nil {
		errnie.Error("failed to get tools", "error", err)
		return nil
	}

	for idx := range tools.Len() {
		tool := tools.At(idx)

		schema := utils.GenerateSchema[struct{}]()

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

		// Create function parameter from tool's schema
		toolParam := openai.ChatCompletionToolParam{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(openai.FunctionDefinitionParam{
				Name:        openai.String(name),
				Description: openai.String(description),
				Parameters:  openai.F(schema.(openai.FunctionParameters)), // Type assertion to FunctionParameters
			}),
		}

		toolsOut = append(toolsOut, toolParam)
	}
	openaiParams.Tools = openai.F(toolsOut)

	return toolsOut
}

func (prvdr *OpenAIProvider) buildResponseFormat(
	params *aiCtx.Artifact,
	openaiParams *openai.ChatCompletionNewParams,
) {
	errnie.Debug("provider.buildResponseFormat")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "openai")
		return
	}

	if _, err := params.Process(); err != nil {
		errnie.Error("failed to get process data", "error", err)
		return
	}

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

	buf := make(map[string]any)
	data, err := params.Process()

	if err != nil {
		errnie.Error("failed to get process data", "error", err)
		return
	}

	if data == nil {
		return
	}

	json.Unmarshal(data, &buf)

	var (
		name        string
		description string
		ok          bool
	)

	if name, ok = buf["name"].(string); !ok {
		return
	}

	if description, ok = buf["description"].(string); !ok {
		return
	}

	schemaParam := openai.ResponseFormatJSONSchemaJSONSchemaParam{
		Name:        openai.F(name),
		Description: openai.F(description),
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

type OpenAIEmbedder struct {
	params   *aiCtx.Artifact
	apiKey   string
	endpoint string
	client   *openai.Client
}

func NewOpenAIEmbedder(apiKey string, endpoint string) *OpenAIEmbedder {
	errnie.Debug("provider.NewOpenAIEmbedder")

	return &OpenAIEmbedder{
		params:   &aiCtx.Artifact{},
		apiKey:   apiKey,
		endpoint: endpoint,
		client:   openai.NewClient(option.WithAPIKey(apiKey)),
	}
}

func (embedder *OpenAIEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIEmbedder.Read", "p", string(p))
	return 0, nil
}

func (embedder *OpenAIEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OpenAIEmbedder.Write", "p", string(p))
	return len(p), nil
}

func (embedder *OpenAIEmbedder) Close() error {
	errnie.Debug("provider.OpenAIEmbedder.Close")
	embedder.params = nil
	return nil
}
