/*
Package llm provides integrations with various Language Model providers.
This package implements the core.LLMProvider interface for different providers
like Anthropic, OpenAI, and others, as well as utility providers like BalancedProvider.
*/
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

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
	Client *http.Client
	/* Tools is a list of tools available to the model */
	Tools []core.Tool
}

// AnthropicToolCall represents a tool call from Anthropic Claude
type AnthropicToolCall struct {
	Name  string                 `json:"name"`
	Input map[string]interface{} `json:"input"`
}

// AnthropicContent represents a content block in Anthropic response
type AnthropicContent struct {
	Type     string             `json:"type"`
	Text     string             `json:"text,omitempty"`
	ToolCall *AnthropicToolCall `json:"tool_call,omitempty"`
}

// AnthropicResponse represents the response structure from Anthropic API
type AnthropicResponse struct {
	Content []AnthropicContent `json:"content"`
}

/*
NewAnthropicProvider creates a new Anthropic provider with the specified API key and model.
It retrieves the base URL from configuration or uses the default Anthropic API endpoint.

Parameters:
  - apiKey: The authentication key for accessing the Anthropic API
  - model: The specific Claude model to use (e.g., "claude-2")
  - tools: Optional list of tools to make available to the model

Returns:
  - A pointer to the initialized AnthropicProvider
*/
func NewAnthropicProvider(apiKey string, model string, tools ...core.Tool) *AnthropicProvider {
	baseURL := viper.GetString("endpoints.anthropic")
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1"
	}

	return &AnthropicProvider{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: baseURL,
		Client:  &http.Client{},
		Tools:   tools,
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

// convertToolsToAnthropicFormat converts core.Tool to Anthropic's tool format
func convertToolsToAnthropicFormat(tools []core.Tool) ([]map[string]interface{}, error) {
	result := make([]map[string]interface{}, 0, len(tools))

	for _, tool := range tools {
		schema := tool.Schema()
		if schema == nil {
			continue
		}

		toolDef := map[string]interface{}{
			"name":         tool.Name(),
			"description":  tool.Description(),
			"input_schema": schema,
		}

		result = append(result, toolDef)
	}

	return result, nil
}

/*
GenerateResponse generates a response from the Anthropic Claude model.
It formats the request according to Anthropic's API requirements, sends the request,
and parses the response.

Parameters:
  - ctx: The context for the request, which can be used for cancellation
  - prompt: The user input to send to the model
  - options: Configuration options for the generation process

Returns:
  - The generated text response
  - An error if the request fails or the response cannot be parsed
*/
func (p *AnthropicProvider) GenerateResponse(ctx context.Context, prompt string, options core.LLMOptions) (string, error) {
	requestMap := map[string]interface{}{
		"model":       p.Model,
		"max_tokens":  options.MaxTokens,
		"temperature": options.Temperature,
		"system":      options.SystemPrompt,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": false,
	}

	// Add tools if available
	if len(p.Tools) > 0 {
		toolDefs, err := convertToolsToAnthropicFormat(p.Tools)
		if err != nil {
			return "", fmt.Errorf("failed to convert tools: %w", err)
		}
		requestMap["tools"] = toolDefs
	}

	requestBody, err := json.Marshal(requestMap)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/messages", bytes.NewBuffer(requestBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", p.APIKey)
	req.Header.Set("Anthropic-Version", "2023-06-01")

	resp, err := p.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Anthropic API error: %s, status: %d", string(body), resp.StatusCode)
	}

	var response AnthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	// Process content - can include text and tool calls
	var result strings.Builder
	toolCalls := make([]core.ToolCall, 0)

	for _, content := range response.Content {
		if content.Type == "text" {
			result.WriteString(content.Text)
		} else if content.Type == "tool_call" && content.ToolCall != nil {
			toolCalls = append(toolCalls, core.ToolCall{
				Name: content.ToolCall.Name,
				Args: content.ToolCall.Input,
			})
		}
	}

	// If we have tool calls, return them as JSON
	if len(toolCalls) > 0 {
		toolCallsJSON, err := json.Marshal(toolCalls)
		if err != nil {
			return "", fmt.Errorf("failed to marshal tool calls: %w", err)
		}
		return string(toolCallsJSON), nil
	}

	return result.String(), nil
}

/*
StreamResponse generates a response from the Anthropic Claude model and streams it.
It sets up a streaming request to the Anthropic API and calls the provided handler
function with each chunk of generated text as it becomes available.

Parameters:
  - ctx: The context for the request, which can be used for cancellation
  - prompt: The user input to send to the model
  - options: Configuration options for the generation process
  - handler: A callback function that receives each text chunk as it's generated

Returns:
  - An error if the request fails or the streaming process encounters an issue
*/
func (p *AnthropicProvider) StreamResponse(ctx context.Context, prompt string, options core.LLMOptions, handler func(string)) error {
	requestMap := map[string]interface{}{
		"model":       p.Model,
		"max_tokens":  options.MaxTokens,
		"temperature": options.Temperature,
		"system":      options.SystemPrompt,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": true,
	}

	// Add tools if available
	if len(p.Tools) > 0 {
		toolDefs, err := convertToolsToAnthropicFormat(p.Tools)
		if err != nil {
			return fmt.Errorf("failed to convert tools: %w", err)
		}
		requestMap["tools"] = toolDefs
	}

	requestBody, err := json.Marshal(requestMap)
	if err != nil {
		return fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/messages", bytes.NewBuffer(requestBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", p.APIKey)
	req.Header.Set("Anthropic-Version", "2023-06-01")

	resp, err := p.Client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Anthropic API error: %s, status: %d", string(body), resp.StatusCode)
	}

	// Track tool calls in progress
	var toolCallInProgress bool
	var toolCallBuffer strings.Builder
	var currentToolName string

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("error reading stream: %w", err)
		}

		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var streamResponse struct {
			Type  string `json:"type"`
			Delta struct {
				Type     string `json:"type"`
				Text     string `json:"text,omitempty"`
				ToolCall struct {
					Name  string          `json:"name,omitempty"`
					Input json.RawMessage `json:"input,omitempty"`
				} `json:"tool_call,omitempty"`
			} `json:"delta"`
		}

		if err := json.Unmarshal([]byte(data), &streamResponse); err != nil {
			return fmt.Errorf("failed to unmarshal stream response: %w", err)
		}

		// Handle text chunks
		if streamResponse.Type == "content_block_delta" && streamResponse.Delta.Type == "text" {
			handler(streamResponse.Delta.Text)
		}

		// Handle tool call chunks
		if streamResponse.Type == "content_block_delta" && streamResponse.Delta.Type == "tool_call" {
			if !toolCallInProgress && streamResponse.Delta.ToolCall.Name != "" {
				// Beginning of a new tool call
				toolCallInProgress = true
				currentToolName = streamResponse.Delta.ToolCall.Name
				toolCallBuffer.Reset() // Clear buffer for new tool call
				handler(fmt.Sprintf("\n[Tool Call: %s", currentToolName))
			}

			// Accumulate the tool call data
			if len(streamResponse.Delta.ToolCall.Input) > 0 {
				toolCallBuffer.Write(streamResponse.Delta.ToolCall.Input)
			}
		}

		// Handle content block start/stop for tool calls
		if streamResponse.Type == "content_block_start" && streamResponse.Delta.Type == "tool_call" {
			toolCallInProgress = true
			if streamResponse.Delta.ToolCall.Name != "" {
				currentToolName = streamResponse.Delta.ToolCall.Name
				handler(fmt.Sprintf("\n[Tool Call: %s", currentToolName))
			}
		}

		if streamResponse.Type == "content_block_stop" && toolCallInProgress {
			// End of tool call - process the accumulated data
			toolCallInProgress = false

			// Parse the accumulated JSON input
			if toolCallBuffer.Len() > 0 {
				var toolArgs map[string]interface{}
				if err := json.Unmarshal([]byte(toolCallBuffer.String()), &toolArgs); err != nil {
					// If we can't parse it, at least show the raw data
					handler(fmt.Sprintf(" with args: %s]\n", toolCallBuffer.String()))
				} else {
					// Format nicely if we can parse it
					handler("]\n")

					// Convert to a structured tool call
					toolCall := core.ToolCall{
						Name: currentToolName,
						Args: toolArgs,
					}

					// You could return this tool call for immediate processing, but for now
					// we'll just indicate that a tool call is ready
					toolCallJSON, _ := json.Marshal(toolCall)
					handler(fmt.Sprintf("[Tool call ready: %s]\n", string(toolCallJSON)))
				}
			} else {
				handler("]\n") // Close the tool call bracket
			}
		}
	}

	// Process any remaining tool call data
	if toolCallInProgress && toolCallBuffer.Len() > 0 {
		var toolArgs map[string]interface{}
		if err := json.Unmarshal([]byte(toolCallBuffer.String()), &toolArgs); err != nil {
			handler(fmt.Sprintf(" with args: %s]\n", toolCallBuffer.String()))
		} else {
			handler("]\n")

			toolCall := core.ToolCall{
				Name: currentToolName,
				Args: toolArgs,
			}

			toolCallJSON, _ := json.Marshal(toolCall)
			handler(fmt.Sprintf("[Tool call ready: %s]\n", string(toolCallJSON)))
		}
	}

	return nil
}
