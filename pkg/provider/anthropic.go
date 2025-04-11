package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
AnthropicProvider implements an LLM provider that connects to Anthropic's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type AnthropicProvider struct {
	client *anthropic.Client
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewAnthropicProvider creates a new Anthropic provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the ANTHROPIC_API_KEY environment variable.
*/
func NewAnthropicProvider(opts ...AnthropicProviderOption) *AnthropicProvider {
	errnie.Debug("provider.NewAnthropicProvider")

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	ctx, cancel := context.WithCancel(context.Background())

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)

	prvdr := &AnthropicProvider{
		client: &client,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *AnthropicProvider) ID() string {
	return "anthropic"
}

func (prvdr *AnthropicProvider) Generate(
	params ProviderParams,
) (ProviderEvent, error) {
	errnie.Info("provider.Generate", "supplier", "anthropic")

	composed := &anthropic.MessageNewParams{
		Model:       anthropic.Model(params.Model),
		Temperature: anthropic.Float(params.Temperature),
		TopP:        anthropic.Float(params.TopP),
	}

	if params.MaxTokens > 1 {
		composed.MaxTokens = int64(params.MaxTokens)
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

func (prvdr *AnthropicProvider) Name() string {
	return "anthropic"
}

type AnthropicProviderOption func(*AnthropicProvider)

func WithAnthropicAPIKey(apiKey string) AnthropicProviderOption {
	return func(provider *AnthropicProvider) {
		provider.client.Options = append(provider.client.Options, option.WithAPIKey(apiKey))
	}
}

func WithAnthropicEndpoint(endpoint string) AnthropicProviderOption {
	return func(provider *AnthropicProvider) {
		provider.client.Options = append(provider.client.Options, option.WithBaseURL(endpoint))
	}
}

func (prvdr *AnthropicProvider) handleSingleRequest(
	params *anthropic.MessageNewParams,
) (ProviderEvent, error) {
	errnie.Debug("provider.handleSingleRequest")

	response, err := prvdr.client.Messages.New(prvdr.ctx, *params)
	if err != nil {
		return ProviderEvent{}, errnie.Error(err)
	}

	if response.Content == nil {
		return ProviderEvent{}, errnie.Error(errors.New("content is nil"))
	}

	var content string
	var toolCalls []anthropic.ToolUseBlock

	for _, block := range response.Content {
		switch block := block.AsAny().(type) {
		case anthropic.TextBlock:
			content += block.Text
		case anthropic.ToolUseBlock:
			toolCalls = append(toolCalls, block)
		}
	}

	// Abort early if there are no tool calls
	if len(toolCalls) == 0 {
		return ProviderEvent{
			Message: Message{
				Role:    "assistant",
				Name:    string(params.Model),
				Content: content,
			},
		}, nil
	}

	// Create tool calls list
	var toolCallsList []mcp.CallToolRequest

	for _, toolCall := range toolCalls {
		errnie.Info("toolCall", "tool", toolCall.Name, "id", toolCall.ID)

		tc := mcp.CallToolRequest{
			Params: struct {
				Name      string                 `json:"name"`
				Arguments map[string]interface{} `json:"arguments,omitempty"`
				Meta      *struct {
					ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
				} `json:"_meta,omitempty"`
			}{
				Name: toolCall.Name,
			},
		}

		// Parse arguments from raw JSON
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(toolCall.JSON.Input.Raw()), &args); err == nil {
			tc.Params.Arguments = args
		}

		toolCallsList = append(toolCallsList, tc)
	}

	// Create artifact with message content
	return ProviderEvent{
		Message: Message{
			Role:      "assistant",
			Name:      string(params.Model),
			Content:   content,
			ToolCalls: toolCallsList,
		},
	}, nil
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *AnthropicProvider) handleStreamingRequest(
	params *anthropic.MessageNewParams,
) (ProviderEvent, error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream := prvdr.client.Messages.NewStreaming(prvdr.ctx, *params)
	defer stream.Close()

	accumulatedMessage := anthropic.Message{}

	for stream.Next() {
		chunk := stream.Current()
		if err := accumulatedMessage.Accumulate(chunk); err != nil {
			return ProviderEvent{}, errnie.Error(err)
		}

		switch event := chunk.AsAny().(type) {
		case anthropic.ContentBlockStartEvent:
			if event.ContentBlock.Name != "" {
				return ProviderEvent{
					Message: Message{
						Role:    "assistant",
						Name:    string(params.Model),
						Content: event.ContentBlock.Name + ": ",
					},
				}, nil
			}
		case anthropic.ContentBlockDeltaEvent:
			if event.Delta.Text != "" {
				return ProviderEvent{
					Message: Message{
						Role:    "assistant",
						Name:    string(params.Model),
						Content: event.Delta.Text,
					},
				}, nil
			}
			if event.Delta.PartialJSON != "" {
				return ProviderEvent{
					Message: Message{
						Role:    "assistant",
						Name:    string(params.Model),
						Content: event.Delta.PartialJSON,
					},
				}, nil
			}
		case anthropic.ContentBlockStopEvent:
			return ProviderEvent{
				Message: Message{
					Role:    "assistant",
					Name:    string(params.Model),
					Content: "\n\n",
				},
			}, nil
		case anthropic.MessageStopEvent:
			return ProviderEvent{
				Message: Message{
					Role:    "assistant",
					Name:    string(params.Model),
					Content: "\n",
				},
			}, nil
		}

		// Handle tool calls if present in the accumulated message
		if len(accumulatedMessage.Content) > 0 {
			for _, block := range accumulatedMessage.Content {
				if block.Type == "tool_use" {
					toolData, err := json.Marshal(block)
					if err != nil {
						errnie.Error("failed to marshal tool_use block", "error", err)
						continue
					}

					var toolInfo struct {
						ID    string                 `json:"id"`
						Name  string                 `json:"name"`
						Input map[string]interface{} `json:"input"`
					}

					if err := json.Unmarshal(toolData, &toolInfo); err != nil {
						errnie.Error("failed to unmarshal tool data", "error", err)
						continue
					}

					inputJSON, err := json.Marshal(toolInfo.Input)
					if err != nil {
						errnie.Error("failed to marshal tool input", "error", err)
						continue
					}

					// Create tool request
					tc := mcp.CallToolRequest{
						Params: struct {
							Name      string                 `json:"name"`
							Arguments map[string]interface{} `json:"arguments,omitempty"`
							Meta      *struct {
								ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
							} `json:"_meta,omitempty"`
						}{
							Name:      toolInfo.Name,
							Arguments: toolInfo.Input,
						},
					}

					errnie.Info("toolCall detected (streaming)", "name", toolInfo.Name)
					return ProviderEvent{
						Message: Message{
							Role:      "assistant",
							Name:      string(params.Model),
							Content:   string(inputJSON),
							ToolCalls: []mcp.CallToolRequest{tc},
						},
					}, nil
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		return ProviderEvent{}, errnie.Error("Streaming error", "error", err)
	}

	return ProviderEvent{}, nil
}

func (prvdr *AnthropicProvider) buildMessages(
	messageParams *anthropic.MessageNewParams,
	messages []Message,
) (err error) {
	errnie.Debug("provider.buildMessages")

	msgParams := make([]anthropic.MessageParam, 0, len(messages))
	var systemMessage string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemMessage = msg.Content
		case "user":
			msgParams = append(msgParams, anthropic.NewUserMessage(anthropic.NewTextBlock(msg.Content)))
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				// Create assistant message with text content
				msgParams = append(msgParams, anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.Content)))

				// Add tool calls information
				for _, toolCall := range msg.ToolCalls {
					toolNote := fmt.Sprintf("[Tool Call: %s, Arguments: %s]", toolCall.Params.Name, toolCall.Params.Arguments)
					msgParams = append(msgParams, anthropic.NewAssistantMessage(anthropic.NewTextBlock(toolNote)))
				}
			} else {
				// Regular assistant message without tool calls
				msgParams = append(msgParams, anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.Content)))
			}
		case "tool":
			// Create a tool result message
			toolMsg := anthropic.NewUserMessage(
				anthropic.NewTextBlock(fmt.Sprintf("[Tool Result from %s: %s]", msg.ID, msg.Content)),
			)

			msgParams = append(msgParams, toolMsg)
		default:
			errnie.Error("unknown message role", "role", msg.Role)
		}
	}

	// Set system message if present
	if systemMessage != "" {
		messageParams.System = []anthropic.TextBlockParam{
			{Text: systemMessage},
		}
	}

	messageParams.Messages = msgParams
	return nil
}

func (prvdr *AnthropicProvider) buildTools(
	messageParams *anthropic.MessageNewParams,
	tools []mcp.Tool,
) (err error) {
	errnie.Debug("provider.buildTools")

	// If no tools, skip
	if len(tools) == 0 {
		return nil
	}

	// Prepare the tools
	toolParams := make([]anthropic.ToolParam, 0, len(tools))

	for _, tool := range tools {
		// Create a tool parameter with this schema
		toolParam := anthropic.ToolParam{
			Name:        tool.Name,
			Description: param.NewOpt(tool.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Type:       "object",
				Properties: tool.InputSchema.Properties,
			},
		}

		toolParams = append(toolParams, toolParam)
	}

	// Set the tools
	toolUnionParams := make([]anthropic.ToolUnionParam, 0, len(toolParams))
	for _, tool := range toolParams {
		toolUnionParams = append(toolUnionParams, anthropic.ToolUnionParam{
			OfTool: &tool,
		})
	}
	messageParams.Tools = toolUnionParams
	return nil
}

func (prvdr *AnthropicProvider) buildResponseFormat(
	messageParams *anthropic.MessageNewParams,
	format ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	// If no format specified, skip
	if format.Name == "" && format.Description == "" && format.Schema == nil {
		return nil
	}

	messageParams.Messages = append(
		messageParams.Messages,
		anthropic.NewAssistantMessage(
			anthropic.NewTextBlock(
				strings.Join([]string{
					"Format your response as a JSON object using the following schema.",
					fmt.Sprintf("Schema:\n\n%v", format.Schema),
					"Strictly follow the schema. Do not leave out required fields, and do not include any non-existent fields or properties.",
					"Output only the JSON object, nothing else, and no Markdown code block.",
				}, "\n\n"),
			),
		),
	)

	return nil
}

type AnthropicEmbedder struct {
	client *anthropic.Client
	ctx    context.Context
	cancel context.CancelFunc
}

func NewAnthropicEmbedder(opts ...AnthropicEmbedderOption) *AnthropicEmbedder {
	errnie.Debug("provider.NewAnthropicEmbedder")

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	ctx, cancel := context.WithCancel(context.Background())

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)

	embedder := &AnthropicEmbedder{
		client: &client,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(embedder)
	}

	return embedder
}

func (embedder *AnthropicEmbedder) Generate(artifact *datura.Artifact) *datura.Artifact {
	errnie.Warn("provider.AnthropicEmbedder.Generate not implemented")
	return datura.New(datura.WithError(errnie.Error(errors.New("embeddings not supported for Anthropic yet"))))
}

type AnthropicEmbedderOption func(*AnthropicEmbedder)

func WithAnthropicEmbedderAPIKey(apiKey string) AnthropicEmbedderOption {
	return func(embedder *AnthropicEmbedder) {
		embedder.client.Options = append(embedder.client.Options, option.WithAPIKey(apiKey))
	}
}

func WithAnthropicEmbedderEndpoint(endpoint string) AnthropicEmbedderOption {
	return func(embedder *AnthropicEmbedder) {
		embedder.client.Options = append(embedder.client.Options, option.WithBaseURL(endpoint))
	}
}
