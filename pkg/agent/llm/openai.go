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

// OpenAIProvider implements the LLMProvider interface for OpenAI
type OpenAIProvider struct {
	APIKey  string
	Model   string
	BaseURL string
	Client  *http.Client
	Tools   []core.Tool
}

// OpenAIToolCall represents a tool call from OpenAI
type OpenAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// OpenAIResponse represents the response structure from OpenAI API
type OpenAIResponse struct {
	Choices []struct {
		Message struct {
			Content   string           `json:"content"`
			ToolCalls []OpenAIToolCall `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
}

// NewOpenAIProvider creates a new OpenAI provider
func NewOpenAIProvider(apiKey string, model string, tools ...core.Tool) *OpenAIProvider {
	baseURL := viper.GetString("endpoints.openai")
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	return &OpenAIProvider{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: baseURL,
		Client:  &http.Client{},
		Tools:   tools,
	}
}

// Name returns the name of the LLM provider
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// GenerateResponse generates a response from the LLM
func (p *OpenAIProvider) GenerateResponse(ctx context.Context, prompt string, options core.LLMOptions) (string, error) {
	messages := []map[string]string{
		{"role": "user", "content": prompt},
	}

	if options.SystemPrompt != "" {
		messages = append([]map[string]string{
			{"role": "system", "content": options.SystemPrompt},
		}, messages...)
	}

	requestBody := map[string]interface{}{
		"model":             p.Model,
		"messages":          messages,
		"max_tokens":        options.MaxTokens,
		"temperature":       options.Temperature,
		"top_p":             options.TopP,
		"presence_penalty":  options.PresencePenalty,
		"frequency_penalty": options.FrequencyPenalty,
		"stop":              options.StopSequences,
		"stream":            false,
	}

	// Add tools if available
	if len(p.Tools) > 0 {
		toolDefs, err := convertToolsToOpenAIFormat(p.Tools)
		if err != nil {
			return "", fmt.Errorf("failed to convert tools: %w", err)
		}
		requestBody["tools"] = toolDefs
		requestBody["tool_choice"] = "auto"
	}

	requestBodyJSON, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/chat/completions", bytes.NewBuffer(requestBodyJSON))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)

	resp, err := p.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("OpenAI API error: %s, status: %d", string(body), resp.StatusCode)
	}

	var response OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no response choices returned")
	}

	// Process tool calls
	if len(response.Choices) > 0 && len(response.Choices[0].Message.ToolCalls) > 0 {
		// Format the tool calls for the agent
		toolCalls := make([]core.ToolCall, 0, len(response.Choices[0].Message.ToolCalls))

		for _, tc := range response.Choices[0].Message.ToolCalls {
			// Parse the arguments JSON string
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				return "", fmt.Errorf("failed to parse tool arguments: %w", err)
			}

			toolCalls = append(toolCalls, core.ToolCall{
				Name: tc.Function.Name,
				Args: args,
			})
		}

		// Return formatted tool calls JSON alongside or instead of content
		toolCallsJSON, err := json.Marshal(toolCalls)
		if err != nil {
			return "", fmt.Errorf("failed to marshal tool calls: %w", err)
		}

		return string(toolCallsJSON), nil
	}

	// Return regular content if no tool calls
	return response.Choices[0].Message.Content, nil
}

// StreamResponse generates a response from the LLM and streams it
func (p *OpenAIProvider) StreamResponse(ctx context.Context, prompt string, options core.LLMOptions, handler func(string)) error {
	messages := []map[string]string{
		{"role": "user", "content": prompt},
	}

	if options.SystemPrompt != "" {
		messages = append([]map[string]string{
			{"role": "system", "content": options.SystemPrompt},
		}, messages...)
	}

	requestBody := map[string]interface{}{
		"model":             p.Model,
		"messages":          messages,
		"max_tokens":        options.MaxTokens,
		"temperature":       options.Temperature,
		"top_p":             options.TopP,
		"presence_penalty":  options.PresencePenalty,
		"frequency_penalty": options.FrequencyPenalty,
		"stop":              options.StopSequences,
		"stream":            true,
	}

	// Add tools if available
	if len(p.Tools) > 0 {
		toolDefs, err := convertToolsToOpenAIFormat(p.Tools)
		if err != nil {
			return fmt.Errorf("failed to convert tools: %w", err)
		}
		requestBody["tools"] = toolDefs
		requestBody["tool_choice"] = "auto"
	}

	requestBodyJSON, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/chat/completions", bytes.NewBuffer(requestBodyJSON))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)

	resp, err := p.Client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("OpenAI API error: %s, status: %d", string(body), resp.StatusCode)
	}

	// For tracking tool calls across chunks
	toolCallsInProgress := make(map[string]*OpenAIToolCall)
	toolCallsComplete := make(map[string]bool)

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
		if line == "" || line == "data: [DONE]" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		var streamResponse struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
		}

		if err := json.Unmarshal([]byte(data), &streamResponse); err != nil {
			return fmt.Errorf("failed to unmarshal stream response: %w", err)
		}

		// Process text content
		if len(streamResponse.Choices) > 0 && streamResponse.Choices[0].Delta.Content != "" {
			handler(streamResponse.Choices[0].Delta.Content)
		}

		// Process tool calls
		if len(streamResponse.Choices) > 0 && len(streamResponse.Choices[0].Delta.ToolCalls) > 0 {
			for _, tc := range streamResponse.Choices[0].Delta.ToolCalls {
				// First chunk of a tool call
				if toolCallsInProgress[tc.ID] == nil && tc.Function.Name != "" {
					toolCallsInProgress[tc.ID] = &OpenAIToolCall{
						ID:   tc.ID,
						Type: tc.Type,
					}
					toolCallsInProgress[tc.ID].Function.Name = tc.Function.Name
					toolCallsInProgress[tc.ID].Function.Arguments = tc.Function.Arguments

					// Signal the start of a tool call to the user
					handler(fmt.Sprintf("\n[Tool Call: %s", tc.Function.Name))
				} else if toolCallsInProgress[tc.ID] != nil {
					// Accumulate arguments
					if tc.Function.Arguments != "" {
						toolCallsInProgress[tc.ID].Function.Arguments += tc.Function.Arguments
					}
				}

				// Check if finished
				if streamResponse.Choices[0].FinishReason == "tool_calls" {
					toolCallsComplete[tc.ID] = true
					handler("]\n")
				}
			}
		}
	}

	// Process any completed tool calls after the stream ends
	if len(toolCallsComplete) > 0 {
		var completedToolCalls []core.ToolCall

		for id, complete := range toolCallsComplete {
			if !complete {
				continue
			}

			tc := toolCallsInProgress[id]
			if tc == nil {
				continue
			}

			// Parse arguments
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				// Just log error and continue
				handler(fmt.Sprintf("\n[Error parsing tool arguments: %v]\n", err))
				continue
			}

			completedToolCalls = append(completedToolCalls, core.ToolCall{
				Name: tc.Function.Name,
				Args: args,
			})
		}

		// Log the tool calls
		handler(fmt.Sprintf("\n[%d tool calls completed]\n", len(completedToolCalls)))
	}

	return nil
}

func convertToolsToOpenAIFormat(tools []core.Tool) ([]map[string]interface{}, error) {
	result := make([]map[string]interface{}, 0, len(tools))

	for _, tool := range tools {
		schema := tool.Schema()
		if schema == nil {
			continue
		}

		toolDef := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        tool.Name(),
				"description": tool.Description(),
				"parameters":  schema,
			},
		}

		result = append(result, toolDef)
	}

	return result, nil
}
