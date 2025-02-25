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
}

// NewOpenAIProvider creates a new OpenAI provider
func NewOpenAIProvider(apiKey string, model string) *OpenAIProvider {
	baseURL := viper.GetString("endpoints.openai")
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	return &OpenAIProvider{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: baseURL,
		Client:  &http.Client{},
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

	requestBody, err := json.Marshal(map[string]interface{}{
		"model":             p.Model,
		"messages":          messages,
		"max_tokens":        options.MaxTokens,
		"temperature":       options.Temperature,
		"top_p":             options.TopP,
		"presence_penalty":  options.PresencePenalty,
		"frequency_penalty": options.FrequencyPenalty,
		"stop":              options.StopSequences,
		"stream":            false,
	})
	if err != nil {
		return "", fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/chat/completions", bytes.NewBuffer(requestBody))
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

	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no response choices returned")
	}

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

	requestBody, err := json.Marshal(map[string]interface{}{
		"model":             p.Model,
		"messages":          messages,
		"max_tokens":        options.MaxTokens,
		"temperature":       options.Temperature,
		"top_p":             options.TopP,
		"presence_penalty":  options.PresencePenalty,
		"frequency_penalty": options.FrequencyPenalty,
		"stop":              options.StopSequences,
		"stream":            true,
	})
	if err != nil {
		return fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/chat/completions", bytes.NewBuffer(requestBody))
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
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}

		if err := json.Unmarshal([]byte(data), &streamResponse); err != nil {
			return fmt.Errorf("failed to unmarshal stream response: %w", err)
		}

		if len(streamResponse.Choices) > 0 && streamResponse.Choices[0].Delta.Content != "" {
			handler(streamResponse.Choices[0].Delta.Content)
		}
	}

	return nil
}
