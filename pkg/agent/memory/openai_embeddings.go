package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OpenAIEmbeddingProvider implements the EmbeddingProvider interface
// using OpenAI's embedding API.
type OpenAIEmbeddingProvider struct {
	// apiKey is the authentication key for the OpenAI API
	apiKey string
	// model is the specific embedding model to use
	model string
}

// NewOpenAIEmbeddingProvider creates a new OpenAI embedding provider.
//
// Parameters:
//   - apiKey: The OpenAI API key
//   - model: The specific model to use (defaults to "text-embedding-3-large" if empty)
//
// Returns:
//   - A pointer to an initialized OpenAIEmbeddingProvider
func NewOpenAIEmbeddingProvider(apiKey, model string) *OpenAIEmbeddingProvider {
	if model == "" {
		model = "text-embedding-3-large"
	}

	return &OpenAIEmbeddingProvider{
		apiKey: apiKey,
		model:  model,
	}
}

// GetEmbedding converts text to vector embeddings using OpenAI's API.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - text: The text to create an embedding for
//
// Returns:
//   - The vector embedding representation
//   - An error if the operation fails, or nil on success
func (o *OpenAIEmbeddingProvider) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	if text == "" {
		return []float32{}, nil
	}

	type EmbeddingRequest struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}

	type EmbeddingData struct {
		Embedding []float32 `json:"embedding"`
	}

	type EmbeddingResponse struct {
		Data []EmbeddingData `json:"data"`
	}

	// Prepare the request
	reqBody := EmbeddingRequest{
		Model: o.model,
		Input: []string{text},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	// Execute the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute HTTP request: %w", err)
	}
	defer resp.Body.Close()

	// Parse the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: %s, status code: %d", string(body), resp.StatusCode)
	}

	var response EmbeddingResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(response.Data) == 0 {
		return nil, fmt.Errorf("no embedding data returned")
	}

	return response.Data[0].Embedding, nil
}
