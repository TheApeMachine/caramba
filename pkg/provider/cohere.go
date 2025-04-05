package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"capnproto.org/go/capnp/v3"
	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/spf13/viper"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/message"
	"github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
CohereProvider implements an LLM provider that connects to Cohere's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type CohereProvider struct {
	client   *cohereclient.Client
	endpoint string
	pctx     context.Context
	ctx      context.Context
	cancel   context.CancelFunc
	segment  *capnp.Segment
}

/*
NewCohereProvider creates a new Cohere provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the COHERE_API_KEY environment variable.
*/
func NewCohereProvider(opts ...CohereProviderOption) *CohereProvider {
	errnie.Debug("provider.NewCohereProvider")

	apiKey := os.Getenv("COHERE_API_KEY")
	endpoint := viper.GetViper().GetString("endpoints.cohere")
	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &CohereProvider{
		client: cohereclient.NewClient(
			cohereclient.WithToken(apiKey),
		),
		endpoint: endpoint,
		pctx:     ctx,
		ctx:      ctx,
		cancel:   cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *CohereProvider) ID() string {
	return "cohere"
}

type CohereProviderOption func(*CohereProvider)

func WithCohereAPIKey(apiKey string) CohereProviderOption {
	return func(prvdr *CohereProvider) {
		prvdr.client = cohereclient.NewClient(
			cohereclient.WithToken(apiKey),
		)
	}
}

func WithCohereEndpoint(endpoint string) CohereProviderOption {
	return func(prvdr *CohereProvider) {
		prvdr.endpoint = endpoint
	}
}

func (prvdr *CohereProvider) Generate(
	params params.Params,
	ctx aicontext.Context,
	tools []mcp.Tool,
) chan *datura.ArtifactBuilder {
	model, err := params.Model()

	out := make(chan *datura.ArtifactBuilder)

	go func() {
		defer close(out)

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		composed := &cohere.ChatStreamRequest{
			Model:            cohere.String(model),
			Temperature:      cohere.Float64(params.Temperature()),
			P:                cohere.Float64(params.TopP()),
			FrequencyPenalty: cohere.Float64(params.FrequencyPenalty()),
			PresencePenalty:  cohere.Float64(params.PresencePenalty()),
		}

		if params.MaxTokens() > 1 {
			composed.MaxTokens = cohere.Int(int(params.MaxTokens()))
		}

		if err = prvdr.buildMessages(composed, ctx); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildTools(composed, tools); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		format, err := params.Format()

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildResponseFormat(composed, format); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if params.Stream() {
			prvdr.handleStreamingRequest(composed, out)
		} else {
			prvdr.handleSingleRequest(composed, out)
		}
	}()

	return out
}

func (prvdr *CohereProvider) Name() string {
	return "cohere"
}

func (prvdr *CohereProvider) handleSingleRequest(
	params *cohere.ChatStreamRequest,
	channel chan *datura.ArtifactBuilder,
) {
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
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	// Create a new message using Cap'n Proto
	msg, err := message.NewMessage(prvdr.segment)
	if errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	// Set message fields
	if err = msg.SetRole("assistant"); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	if err = msg.SetName("cohere"); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	if err = msg.SetContent(response.Text); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	// Check for tool calls
	toolCalls := response.GetToolCalls()

	// Abort early if there are no tool calls
	if len(toolCalls) == 0 {
		channel <- datura.New(datura.WithPayload([]byte(response.Text)))
		return
	}

	// Create tool calls list
	toolCallList, err := msg.NewToolCalls(int32(len(toolCalls)))
	if errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	for i, toolCall := range toolCalls {
		// Cohere's ToolCall has Name, Parameters fields
		name := toolCall.GetName()

		// Generate a simple ID since Cohere doesn't provide one
		id := fmt.Sprintf("tool-%d", i)

		// Marshal parameters to JSON string for arguments
		paramBytes, err := json.Marshal(toolCall.GetParameters())
		if err != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}
		arguments := string(paramBytes)

		errnie.Info("toolCall", "tool", name, "id", id)

		if err = toolCallList.At(i).SetId(id); errnie.Error(err) != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = toolCallList.At(i).SetName(name); errnie.Error(err) != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = toolCallList.At(i).SetArguments(arguments); errnie.Error(err) != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}
	}

	// Create artifact with message content
	channel <- datura.New(datura.WithPayload([]byte(response.Text)))
}

func (prvdr *CohereProvider) handleStreamingRequest(
	params *cohere.ChatStreamRequest,
	channel chan *datura.ArtifactBuilder,
) {
	errnie.Debug("provider.handleStreamingRequest")

	stream, err := prvdr.client.ChatStream(prvdr.ctx, params)
	if errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	defer stream.Close()

	for {
		chunk, err := stream.Recv()

		if err != nil {
			if err == io.EOF {
				break
			}

			channel <- datura.New(datura.WithError(errnie.Error(err)))
			continue
		}

		if content := chunk.TextGeneration.String(); content != "" {
			channel <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte(content)),
			)
		}
	}
}

func (prvdr *CohereProvider) buildMessages(
	chatParams *cohere.ChatStreamRequest,
	ctx aicontext.Context,
) (err error) {
	errnie.Debug("provider.buildMessages")

	msgs, err := ctx.Messages()
	if errnie.Error(err) != nil {
		return err
	}

	messageList := make([]*cohere.Message, 0, msgs.Len())
	var systemMessage string

	for i := 0; i < msgs.Len(); i++ {
		msg := msgs.At(i)

		role, err := msg.Role()
		if errnie.Error(err) != nil {
			return err
		}

		content, err := msg.Content()
		if errnie.Error(err) != nil {
			return err
		}

		switch role {
		case "system":
			systemMessage = content
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

	if systemMessage != "" {
		chatParams.Preamble = cohere.String(systemMessage)
	}

	chatParams.ChatHistory = messageList
	return nil
}

func (prvdr *CohereProvider) buildTools(
	chatParams *cohere.ChatStreamRequest,
	tools []mcp.Tool,
) (err error) {
	errnie.Debug("provider.buildTools")

	if len(tools) == 0 {
		return nil
	}

	toolList := make([]*cohere.Tool, 0, len(tools))

	for _, tool := range tools {
		parameterDefinitions := make(
			map[string]*cohere.ToolParameterDefinitionsValue,
			len(tool.InputSchema.Properties),
		)

		for name, property := range tool.InputSchema.Properties {
			propMap, ok := property.(map[string]interface{})
			if !ok {
				continue
			}

			description, _ := propMap["description"].(string)
			required := false

			// Check if the property is required
			for _, req := range tool.InputSchema.Required {
				if req == name {
					required = true
					break
				}
			}

			parameterDefinitions[name] = &cohere.ToolParameterDefinitionsValue{
				Type:        propMap["type"].(string),
				Description: cohere.String(description),
				Required:    cohere.Bool(required),
			}
		}

		toolList = append(toolList, &cohere.Tool{
			Name:                 tool.Name,
			Description:          tool.Description,
			ParameterDefinitions: parameterDefinitions,
		})
	}

	if len(toolList) > 0 {
		chatParams.Tools = toolList
	}

	return nil
}

func (prvdr *CohereProvider) buildResponseFormat(
	chatParams *cohere.ChatStreamRequest,
	format params.ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	name, err := format.Name()
	if errnie.Error(err) != nil {
		return err
	}

	description, err := format.Description()
	if errnie.Error(err) != nil {
		return err
	}

	schema, err := format.Schema()
	if errnie.Error(err) != nil {
		return err
	}

	// If no format is specified, return early
	if name == "" && description == "" && schema == "" {
		return nil
	}

	var schemaMap map[string]interface{}
	if err = json.Unmarshal([]byte(schema), &schemaMap); err != nil {
		return errnie.Error(err)
	}

	chatParams.ResponseFormat = &cohere.ResponseFormat{
		Type: "json_object",
		JsonObject: &cohere.JsonResponseFormat{
			Schema: schemaMap,
		},
	}

	return nil
}

type CohereEmbedder struct {
	client   *cohereclient.Client
	apiKey   string
	endpoint string
	ctx      context.Context
	cancel   context.CancelFunc
}

func NewCohereEmbedder(apiKey string, endpoint string) *CohereEmbedder {
	errnie.Debug("provider.NewCohereEmbedder")

	if apiKey == "" {
		apiKey = os.Getenv("COHERE_API_KEY")
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &CohereEmbedder{
		apiKey:   apiKey,
		endpoint: endpoint,
		client:   cohereclient.NewClient(cohereclient.WithToken(apiKey)),
		ctx:      ctx,
		cancel:   cancel,
	}
}

func (embedder *CohereEmbedder) Generate(
	buffer chan *datura.ArtifactBuilder,
	fn ...func(artifact *datura.ArtifactBuilder) *datura.ArtifactBuilder,
) chan *datura.ArtifactBuilder {
	errnie.Debug("provider.CohereEmbedder.Generate")

	out := make(chan *datura.ArtifactBuilder)

	go func() {
		defer close(out)

		select {
		case <-embedder.ctx.Done():
			errnie.Debug("provider.CohereEmbedder.Generate.ctx.Done")
			embedder.cancel()
			return
		case artifact := <-buffer:
			content, err := artifact.DecryptPayload()
			if err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			if len(content) == 0 {
				out <- datura.New(datura.WithError(errnie.Error(errors.New("content is empty"))))
				return
			}

			in := cohere.EmbedInputType(cohere.EmbedInputTypeSearchDocument)

			embedRequest := cohere.EmbedRequest{
				Texts:     []string{string(content)},
				Model:     cohere.String("embed-english-v3.0"),
				InputType: &in,
			}

			response, err := embedder.client.Embed(context.Background(), &embedRequest)
			if err != nil {
				errnie.Error("embedding request failed", "error", err)
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			if response != nil && len(response.EmbeddingsFloats.Embeddings) > 0 {
				embeddings := response.EmbeddingsFloats.Embeddings
				errnie.Debug("created embeddings",
					"text_length", len(string(content)),
					"dimensions", len(embeddings),
				)

				// Convert to bytes - not fully implemented
				out <- datura.New(datura.WithPayload([]byte(string(content))))
			}
		}
	}()

	return out
}

func (embedder *CohereEmbedder) Close() error {
	errnie.Debug("provider.CohereEmbedder.Close")
	embedder.cancel()
	return nil
}

type CohereEmbedderOption func(*CohereEmbedder)

func WithCohereEmbedderAPIKey(apiKey string) CohereEmbedderOption {
	return func(embedder *CohereEmbedder) {
		embedder.client = cohereclient.NewClient(cohereclient.WithToken(apiKey))
	}
}

func WithCohereEmbedderEndpoint(endpoint string) CohereEmbedderOption {
	return func(embedder *CohereEmbedder) {
		embedder.endpoint = endpoint
	}
}
