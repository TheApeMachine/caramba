/*
Package llm provides integrations with various Language Model providers.
This package implements the core.LLMProvider interface for different providers
like Anthropic, OpenAI, and others, as well as utility providers like BalancedProvider.
*/
package llm

import (
	"context"
	"encoding/json"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/agent/core"
)

/*
AnthropicProvider implements the LLMProvider interface for Anthropic's Claude models.
It handles API authentication, request formatting, and response parsing for both
synchronous and streaming responses from the Anthropic API.
*/
type AnthropicProvider struct {
	/* APIKey is the authentication key for the Anthropic API */
	APIKey string
	/* Model is the specific Claude model to use (e.g., "claude-2") */
	Model string
	/* BaseURL is the endpoint for the Anthropic API */
	BaseURL string
	/* Client is the HTTP client used for API requests */
	Client *anthropic.Client
}

/*
NewAnthropicProvider creates a new Anthropic provider with the specified API key and model.
It retrieves the base URL from configuration or uses the default Anthropic API endpoint.

Parameters:
  - apiKey: The authentication key for accessing the Anthropic API
  - model: The specific Claude model to use (e.g., "claude-2")

Returns:
  - A pointer to the initialized AnthropicProvider
*/
func NewAnthropicProvider(apiKey string, model string) *AnthropicProvider {
	baseURL := viper.GetString("endpoints.anthropic")
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1"
	}

	// Initialize client with API key
	client := anthropic.NewClient(option.WithAPIKey(apiKey))

	return &AnthropicProvider{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: baseURL,
		Client:  client,
	}
}

/*
Name returns the name of the LLM provider.
This is used for identification and logging purposes.

Returns:
  - The string "anthropic"
*/
func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

/*
GenerateResponse generates a response from the Anthropic Claude model.
It formats the request according to Anthropic's API requirements, sends the request,
and parses the response.

Parameters:
  - ctx: The context for the request, which can be used for cancellation
  - params: The parameters for the generation process

Returns:
  - The generated text response
  - An error if the request fails or the response cannot be parsed
*/
func (p *AnthropicProvider) GenerateResponse(
	ctx context.Context,
	params core.LLMParams,
) core.LLMResponse {
	// Build the message parameters
	messageParams := anthropic.MessageNewParams{
		Model:     anthropic.F(p.Model),
		MaxTokens: anthropic.Int(1024),
		Messages:  anthropic.F(p.buildMessages(params)),
	}

	// Set system prompt if available
	systemPrompt := p.extractSystemPrompt(params)
	if systemPrompt != "" {
		// System should be TextBlockParam array
		messageParams.System = anthropic.F([]anthropic.TextBlockParam{
			{Text: anthropic.F(systemPrompt)},
		})
	}

	// Add tools if available
	tools := p.buildTools(params)
	if len(tools) > 0 {
		toolUnionParams := make([]anthropic.ToolUnionUnionParam, 0, len(tools))
		for _, tool := range tools {
			toolUnionParams = append(toolUnionParams, anthropic.ToolUnionUnionParam(tool))
		}
		messageParams.Tools = anthropic.F(toolUnionParams)
	}

	// Make the API call
	response, err := p.Client.Messages.New(ctx, messageParams)
	if err != nil {
		return core.LLMResponse{
			Type:  core.ResponseTypeError,
			Model: p.Model,
			Error: err,
		}
	}

	// Process content and handle tool calls
	var result string
	toolCalls := make([]core.ToolCall, 0)

	for _, block := range response.Content {
		if block.Type == anthropic.ContentBlockTypeText {
			result += block.Text
		} else if block.Type == anthropic.ContentBlockTypeToolUse {
			toolCall := core.ToolCall{
				Name: block.Name,
				Args: p.parseToolInput(block.Input),
			}
			toolCalls = append(toolCalls, toolCall)
		}
	}

	// If we have tool calls, return them as JSON
	if len(toolCalls) > 0 {
		toolCallsJSON, err := json.Marshal(toolCalls)

		if err != nil {
			return core.LLMResponse{
				Type:  core.ResponseTypeError,
				Model: p.Model,
				Error: err,
			}
		}

		return core.LLMResponse{
			Type:      core.ResponseTypeToolCall,
			Model:     p.Model,
			ToolCalls: toolCalls,
			Content:   string(toolCallsJSON),
		}
	}

	return core.LLMResponse{
		Type:      core.ResponseTypeContent,
		Model:     p.Model,
		Content:   result,
		ToolCalls: []core.ToolCall{},
	}
}

/*
StreamResponse generates a response from the Anthropic Claude model and streams it.
It sets up a streaming request to the Anthropic API and calls the provided handler
function with each chunk of generated text as it becomes available.

Parameters:
  - ctx: The context for the request, which can be used for cancellation
  - params: The parameters for the generation process

Returns:
  - A channel that emits LLMResponse objects
*/
func (p *AnthropicProvider) StreamResponse(
	ctx context.Context,
	params core.LLMParams,
) <-chan core.LLMResponse {
	out := make(chan core.LLMResponse)

	go func() {
		defer close(out)

		// Build the message parameters
		messageParams := anthropic.MessageNewParams{
			Model:     anthropic.F(p.Model),
			MaxTokens: anthropic.Int(1024),
			Messages:  anthropic.F(p.buildMessages(params)),
		}

		// Set system prompt if available
		systemPrompt := p.extractSystemPrompt(params)
		if systemPrompt != "" {
			messageParams.System = anthropic.F([]anthropic.TextBlockParam{
				{Text: anthropic.F(systemPrompt)},
			})
		}

		// Add tools if available
		tools := p.buildTools(params)
		if len(tools) > 0 {
			toolUnionParams := make([]anthropic.ToolUnionUnionParam, 0, len(tools))
			for _, tool := range tools {
				toolUnionParams = append(toolUnionParams, anthropic.ToolUnionUnionParam(tool))
			}
			messageParams.Tools = anthropic.F(toolUnionParams)
		}

		// Start streaming
		stream := p.Client.Messages.NewStreaming(ctx, messageParams)

		// Track the full message being built
		var fullMessage anthropic.Message

		// Process streaming events
		for stream.Next() {
			// Get the current event
			event := stream.Current()

			// Accumulate the message data as it comes in
			if err := fullMessage.Accumulate(event); err != nil {
				out <- core.LLMResponse{Error: err}
				return
			}

			// Process different event types
			switch evt := event.AsUnion().(type) {
			case anthropic.ContentBlockStartEvent:
				cb := evt.ContentBlock
				if cb.Type == anthropic.ContentBlockStartEventContentBlockTypeToolUse && cb.Name != "" {
					out <- core.LLMResponse{
						Content: "[Tool Call: " + cb.Name + "]",
					}
				}

			case anthropic.ContentBlockDeltaEvent:
				delta := evt.Delta
				if delta.Type == anthropic.ContentBlockDeltaEventDeltaTypeTextDelta && delta.Text != "" {
					out <- core.LLMResponse{
						Content: delta.Text,
					}
				}

			case anthropic.ContentBlockStopEvent:
				// Find the completed tool call in the full message
				for _, block := range fullMessage.Content {
					if block.Type == anthropic.ContentBlockTypeToolUse {
						toolCall := core.ToolCall{
							Name: block.Name,
							Args: p.parseToolInput(block.Input),
						}
						out <- core.LLMResponse{
							ToolCalls: []core.ToolCall{toolCall},
						}
						break
					}
				}

			case anthropic.MessageStopEvent:
				// Message is complete, nothing to do
			}
		}

		// Check for errors
		if err := stream.Err(); err != nil {
			out <- core.LLMResponse{Error: err}
		}
	}()

	return out
}

func (p *AnthropicProvider) buildMessages(
	params core.LLMParams,
) []anthropic.MessageParam {
	messages := make([]anthropic.MessageParam, 0, len(params.Messages))

	for _, message := range params.Messages {
		// Skip system messages as they're handled separately
		if message.Role == "system" {
			continue
		}

		if message.Role == "user" {
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(message.Content),
			))
		} else if message.Role == "assistant" {
			messages = append(messages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(message.Content),
			))
		}
	}

	return messages
}

func (p *AnthropicProvider) buildTools(
	params core.LLMParams,
) []anthropic.ToolParam {
	tools := make([]anthropic.ToolParam, 0, len(params.Tools))

	for _, tool := range params.Tools {
		toolParam := anthropic.ToolParam{
			Name:        anthropic.F(tool.Name()),
			Description: anthropic.F(tool.Description()),
		}

		// Get the JSON schema for the tool
		if schema := tool.Schema(); schema != nil {
			// Convert schema to interface{} before wrapping with F
			schemaInterface := interface{}(schema)
			toolParam.InputSchema = anthropic.F(schemaInterface)
		}

		tools = append(tools, toolParam)
	}

	return tools
}

// parseToolInput converts the raw JSON input from a tool call into a map
func (p *AnthropicProvider) parseToolInput(input json.RawMessage) map[string]interface{} {
	var result map[string]interface{}
	err := json.Unmarshal(input, &result)
	if err != nil {
		// If we can't parse it, return an empty map
		return map[string]interface{}{}
	}
	return result
}

// extractSystemPrompt gets the system prompt from the messages array if it exists
func (p *AnthropicProvider) extractSystemPrompt(params core.LLMParams) string {
	for _, message := range params.Messages {
		if message.Role == "system" {
			return message.Content
		}
	}
	return ""
}
