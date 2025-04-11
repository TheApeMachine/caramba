package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/spf13/viper"
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
	params ProviderParams,
) (ProviderEvent, error) {
	errnie.Info("provider.Generate", "supplier", "cohere")

	composed := &cohere.ChatStreamRequest{
		Model:            cohere.String(params.Model),
		Temperature:      cohere.Float64(params.Temperature),
		P:                cohere.Float64(params.TopP),
		FrequencyPenalty: cohere.Float64(params.FrequencyPenalty),
		PresencePenalty:  cohere.Float64(params.PresencePenalty),
	}

	if params.MaxTokens > 1 {
		composed.MaxTokens = cohere.Int(int(params.MaxTokens))
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

func (prvdr *CohereProvider) Name() string {
	return "cohere"
}

func (prvdr *CohereProvider) handleSingleRequest(
	params *cohere.ChatStreamRequest,
) (ProviderEvent, error) {
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
		return ProviderEvent{}, errnie.Error(err)
	}

	// Check for tool calls
	toolCalls := response.GetToolCalls()

	// Abort early if there are no tool calls
	if len(toolCalls) == 0 {
		return ProviderEvent{
			Message: Message{
				Role:    "assistant",
				Name:    "cohere",
				Content: response.Text,
			},
		}, nil
	}

	// Create tool calls list
	var toolCallsList []mcp.CallToolRequest

	for i, toolCall := range toolCalls {
		// Cohere's ToolCall has Name, Parameters fields
		name := toolCall.GetName()

		// Generate a simple ID since Cohere doesn't provide one
		id := fmt.Sprintf("tool-%d", i)

		// Marshal parameters to JSON string for arguments
		paramBytes, err := json.Marshal(toolCall.GetParameters())
		if err != nil {
			return ProviderEvent{}, errnie.Error(err)
		}

		errnie.Info("toolCall", "tool", name, "id", id)

		tc := mcp.CallToolRequest{
			Params: struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments,omitempty"`
				Meta      *struct {
					ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
				} `json:"_meta,omitempty"`
			}{
				Name: name,
			},
		}

		// Parse arguments from JSON
		var args map[string]any
		if err := json.Unmarshal(paramBytes, &args); err == nil {
			tc.Params.Arguments = args
		}

		toolCallsList = append(toolCallsList, tc)
	}

	// Return provider event with message and tool calls
	return ProviderEvent{
		Message: Message{
			Role:      "assistant",
			Name:      "cohere",
			Content:   response.Text,
			ToolCalls: toolCallsList,
		},
	}, nil
}

func (prvdr *CohereProvider) handleStreamingRequest(
	params *cohere.ChatStreamRequest,
) (ProviderEvent, error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream, err := prvdr.client.ChatStream(prvdr.ctx, params)
	if errnie.Error(err) != nil {
		return ProviderEvent{}, errnie.Error(err)
	}

	defer stream.Close()

	for {
		chunk, err := stream.Recv()

		if err != nil {
			if err == io.EOF {
				break
			}

			return ProviderEvent{}, errnie.Error(err)
		}

		if content := chunk.TextGeneration.String(); content != "" {
			return ProviderEvent{
				Message: Message{
					Role:    "assistant",
					Name:    "cohere",
					Content: content,
				},
			}, nil
		}
	}

	return ProviderEvent{}, nil
}

func (prvdr *CohereProvider) buildMessages(
	chatParams *cohere.ChatStreamRequest,
	messages []Message,
) (err error) {
	errnie.Debug("provider.buildMessages")

	messageList := make([]*cohere.Message, 0, len(messages))
	var systemMessage string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemMessage = msg.Content
		case "user":
			messageList = append(messageList, &cohere.Message{
				Role: "user",
				User: &cohere.ChatMessage{
					Message: msg.Content,
				},
			})
		case "assistant":
			messageList = append(messageList, &cohere.Message{
				Role: "chatbot",
				Chatbot: &cohere.ChatMessage{
					Message: msg.Content,
				},
			})
		default:
			errnie.Error("unknown message role", "role", msg.Role)
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
			propMap, ok := property.(map[string]any)
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
	format ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	// If no format is specified, return early
	if format.Name == "" && format.Description == "" && format.Schema == nil {
		return nil
	}

	var schemaMap map[string]any
	schemaStr, ok := format.Schema.(string)
	if ok {
		if err = json.Unmarshal([]byte(schemaStr), &schemaMap); err != nil {
			return errnie.Error(err)
		}
	} else {
		// Try to use the schema directly if it's already a map
		schemaMap, ok = format.Schema.(map[string]any)
		if !ok {
			return errnie.Error(errors.New("schema is not a string or map"))
		}
	}

	chatParams.ResponseFormat = &cohere.ResponseFormat{
		Type: "json_object",
		JsonObject: &cohere.JsonResponseFormat{
			Schema: schemaMap,
		},
	}

	return nil
}
