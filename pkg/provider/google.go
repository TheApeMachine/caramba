package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
	"google.golang.org/genai"
)

/*
GoogleProvider implements an LLM provider that connects to Google's Gemini API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type GoogleProvider struct {
	client   *genai.Client
	endpoint string
	ctx      context.Context
	cancel   context.CancelFunc
}

/*
NewGoogleProvider creates a new Google Gemini provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the GOOGLE_API_KEY environment variable.
*/
func NewGoogleProvider(opts ...GoogleProviderOption) *GoogleProvider {
	errnie.Debug("provider.NewGoogleProvider")

	ctx, cancel := context.WithCancel(context.Background())

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  os.Getenv("GOOGLE_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})

	if err != nil {
		errnie.Error("failed to create Google client", "error", err)
		cancel()
		return nil
	}

	prvdr := &GoogleProvider{
		client:   client,
		endpoint: "",
		ctx:      ctx,
		cancel:   cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *GoogleProvider) ID() string {
	return "google"
}

type GoogleProviderOption func(*GoogleProvider)

func WithGoogleAPIKey(apiKey string) GoogleProviderOption {
	return func(prvdr *GoogleProvider) {
		prvdr.client, _ = genai.NewClient(context.Background(), &genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		})
	}
}

func WithGoogleEndpoint(endpoint string) GoogleProviderOption {
	return func(prvdr *GoogleProvider) {
		prvdr.endpoint = endpoint
	}
}

func (prvdr *GoogleProvider) Generate(
	params ProviderParams,
) (ProviderEvent, error) {
	errnie.Info("provider.Generate", "supplier", "google")

	chatConfig := &genai.GenerateContentConfig{
		Temperature:      utils.Ptr(float32(params.Temperature)),
		TopP:             utils.Ptr(float32(params.TopP)),
		FrequencyPenalty: utils.Ptr(float32(params.FrequencyPenalty)),
		PresencePenalty:  utils.Ptr(float32(params.PresencePenalty)),
	}

	if params.MaxTokens > 1 {
		chatConfig.MaxOutputTokens = utils.Ptr(int32(params.MaxTokens))
	}

	messages, err := prvdr.buildMessages(params.Messages)
	if err != nil {
		return ProviderEvent{}, err
	}

	if err = prvdr.buildTools(chatConfig, params.Tools); err != nil {
		return ProviderEvent{}, err
	}

	if params.ResponseFormat != (ResponseFormat{}) {
		if err = prvdr.buildResponseFormat(chatConfig, params.ResponseFormat); err != nil {
			return ProviderEvent{}, err
		}
	}

	if params.Stream {
		return prvdr.handleStreamingRequest(chatConfig, messages, params.Model)
	}

	return prvdr.handleSingleRequest(chatConfig, params.Model, messages)
}

func (prvdr *GoogleProvider) Name() string {
	return "google"
}

func (prvdr *GoogleProvider) handleSingleRequest(
	chatConfig *genai.GenerateContentConfig,
	model string,
	messages []*genai.Content,
) (ProviderEvent, error) {
	errnie.Debug("provider.handleSingleRequest")

	resp, err := prvdr.client.Models.GenerateContent(
		context.Background(),
		model,
		messages,
		chatConfig,
	)

	if err != nil {
		return ProviderEvent{}, errnie.Error(err)
	}

	if len(resp.Candidates) == 0 {
		err = errors.New("no response candidates")
		return ProviderEvent{}, errnie.Error(err)
	}

	contentText := ""
	// Combine all parts into the content
	for _, part := range resp.Candidates[0].Content.Parts {
		contentText += part.Text
	}

	// Check for tool calls
	var toolCalls []mcp.CallToolRequest
	if resp.Candidates[0].Content.Parts[0].FunctionCall != nil {
		fc := resp.Candidates[0].Content.Parts[0].FunctionCall

		tc := mcp.CallToolRequest{
			Params: struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments,omitempty"`
				Meta      *struct {
					ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
				} `json:"_meta,omitempty"`
			}{
				Name:      fc.Name,
				Arguments: fc.Args,
			},
		}

		toolCalls = append(toolCalls, tc)
		errnie.Info("toolCall detected", "name", fc.Name)
	}

	// Return provider event with message
	return ProviderEvent{
		Message: Message{
			Role:      "assistant",
			Name:      model,
			Content:   contentText,
			ToolCalls: toolCalls,
		},
	}, nil
}

func (prvdr *GoogleProvider) handleStreamingRequest(
	chatConfig *genai.GenerateContentConfig,
	messages []*genai.Content,
	model string,
) (ProviderEvent, error) {
	errnie.Debug("provider.handleStreamingRequest")

	var lastContent string

	for response, err := range prvdr.client.Models.GenerateContentStream(
		prvdr.ctx,
		model,
		messages,
		chatConfig,
	) {
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			errnie.Error("streaming error", "error", err)
			continue
		}

		if len(response.Candidates) == 0 {
			continue
		}

		for _, part := range response.Candidates[0].Content.Parts {
			if part.Text != "" {
				lastContent = part.Text
			}

			// Handle tool calls in streaming mode
			if part.FunctionCall != nil {
				fc := part.FunctionCall

				tc := mcp.CallToolRequest{
					Params: struct {
						Name      string         `json:"name"`
						Arguments map[string]any `json:"arguments,omitempty"`
						Meta      *struct {
							ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
						} `json:"_meta,omitempty"`
					}{
						Name:      fc.Name,
						Arguments: fc.Args,
					},
				}

				errnie.Info("toolCall detected (streaming)", "name", fc.Name)

				// Return immediately if we found a tool call
				return ProviderEvent{
					Message: Message{
						Role:      "assistant",
						Name:      model,
						Content:   lastContent,
						ToolCalls: []mcp.CallToolRequest{tc},
					},
				}, nil
			}
		}
	}

	// Return the last content if no tool calls were found
	return ProviderEvent{
		Message: Message{
			Role:    "assistant",
			Name:    model,
			Content: lastContent,
		},
	}, nil
}

func (prvdr *GoogleProvider) buildMessages(
	messages []Message,
) (genaiMessages []*genai.Content, err error) {
	errnie.Debug("provider.buildMessages")

	genaiMessages = make([]*genai.Content, 0, len(messages))

	for _, msg := range messages {
		msgParts := []*genai.Part{{Text: msg.Content}}

		// Handle tool calls for assistant messages
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				// Handle different argument formats
				if toolCall.Params.Arguments != nil {
					// Just use the arguments map directly
					args := toolCall.Params.Arguments

					msgParts = append(msgParts, &genai.Part{
						FunctionCall: &genai.FunctionCall{
							Name: toolCall.Params.Name,
							Args: args,
						},
					})
				}
			}
		}

		content := &genai.Content{
			Role:  msg.Role,
			Parts: msgParts,
		}

		genaiMessages = append(genaiMessages, content)
	}

	return genaiMessages, nil
}

func (prvdr *GoogleProvider) buildTools(
	chatConfig *genai.GenerateContentConfig,
	tools []mcp.Tool,
) (err error) {
	errnie.Debug("provider.buildTools")

	if len(tools) == 0 {
		chatConfig.Tools = nil
		return nil
	}

	googleTools := make([]*genai.Tool, 0, len(tools))

	for _, tool := range tools {
		functionDeclaration := &genai.FunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
		}

		// Convert parameters to Google's format
		properties := tool.InputSchema.Properties
		required := tool.InputSchema.Required

		if len(properties) > 0 {
			schema := &genai.Schema{
				Type:       genai.Type("object"),
				Properties: make(map[string]*genai.Schema),
				Required:   required,
			}

			for propName, propValue := range properties {
				propMap, ok := propValue.(map[string]any)
				if !ok {
					continue
				}

				propType, _ := propMap["type"].(string)
				propDescription, _ := propMap["description"].(string)

				schema.Properties[propName] = &genai.Schema{
					Type:        genai.Type(propType),
					Description: propDescription,
				}
			}

			functionDeclaration.Parameters = schema
		}

		googleTools = append(googleTools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{functionDeclaration},
		})
	}

	chatConfig.Tools = googleTools
	return nil
}

func (prvdr *GoogleProvider) buildResponseFormat(
	chatConfig *genai.GenerateContentConfig,
	format ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	// Check if format is a zero value
	if format.Name == "" && format.Description == "" && format.Schema == nil {
		return nil
	}

	// Google's API doesn't have direct JSON schema support like OpenAI
	// Instead, we'll add it to the system message
	systemMsg := fmt.Sprintf(
		"Please format your response as a JSON object following this schema:\n%s\n%s",
		format.Name,
		format.Description,
	)

	if format.Schema != nil {
		schemaStr := ""
		switch s := format.Schema.(type) {
		case string:
			schemaStr = s
		default:
			j, _ := json.Marshal(format.Schema)
			schemaStr = string(j)
		}
		systemMsg += fmt.Sprintf("\nSchema: %v", schemaStr)
	}

	chatConfig.SystemInstruction = &genai.Content{
		Role:  "system",
		Parts: []*genai.Part{{Text: systemMsg}},
	}

	return nil
}
