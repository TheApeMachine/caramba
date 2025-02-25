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

	return &AnthropicProvider{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: baseURL,
		Client:  &http.Client{},
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
  - prompt: The user input to send to the model
  - options: Configuration options for the generation process

Returns:
  - The generated text response
  - An error if the request fails or the response cannot be parsed
*/
func (p *AnthropicProvider) GenerateResponse(ctx context.Context, prompt string, options core.LLMOptions) (string, error) {
	requestBody, err := json.Marshal(map[string]interface{}{
		"model":       p.Model,
		"max_tokens":  options.MaxTokens,
		"temperature": options.Temperature,
		"system":      options.SystemPrompt,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": false,
	})
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

	var response struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	var result strings.Builder
	for _, content := range response.Content {
		if content.Type == "text" {
			result.WriteString(content.Text)
		}
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
	requestBody, err := json.Marshal(map[string]interface{}{
		"model":       p.Model,
		"max_tokens":  options.MaxTokens,
		"temperature": options.Temperature,
		"system":      options.SystemPrompt,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": true,
	})
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
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
		}

		if err := json.Unmarshal([]byte(data), &streamResponse); err != nil {
			return fmt.Errorf("failed to unmarshal stream response: %w", err)
		}

		if streamResponse.Type == "content_block_delta" && streamResponse.Delta.Type == "text" {
			handler(streamResponse.Delta.Text)
		}
	}

	return nil
}
