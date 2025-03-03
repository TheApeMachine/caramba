package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/theapemachine/caramba/pkg/output"
)

//------------------------------------------------------------------------------
// OpenAI SDK Implementation
//------------------------------------------------------------------------------

// OpenAISDKEmbeddings implements EmbeddingProvider using the official OpenAI SDK
type OpenAISDKEmbeddings struct {
	logger           *output.Logger
	client           *openai.Client
	embeddingService *openai.EmbeddingService
	model            openai.EmbeddingModel
}

// NewOpenAISDKEmbeddings creates a new embedding provider using the official OpenAI SDK
func NewOpenAISDKEmbeddings(apiKey string, model string) *OpenAISDKEmbeddings {
	if model == "" {
		model = openai.EmbeddingModelTextEmbedding3Large
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	return &OpenAISDKEmbeddings{
		logger:           output.NewLogger(),
		client:           client,
		embeddingService: openai.NewEmbeddingService(),
		model:            model,
	}
}

// GetEmbedding implements the EmbeddingProvider interface using the OpenAI SDK
func (o *OpenAISDKEmbeddings) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	if text == "" {
		return nil, fmt.Errorf("cannot get embeddings for empty text")
	}

	// Create input array of strings as required by the API
	inputStrings := []string{text}

	// Call the embedding API with the correct parameters
	response, err := o.embeddingService.New(
		ctx,
		openai.EmbeddingNewParams{
			Input: openai.F[openai.EmbeddingNewParamsInputUnion](openai.EmbeddingNewParamsInputArrayOfStrings(inputStrings)),
			Model: openai.F(o.model),
		},
	)
	if err != nil {
		return nil, o.logger.Error(
			"openai",
			fmt.Errorf("failed to get OpenAI embeddings: %w", err),
		)
	}

	if len(response.Data) == 0 {
		return nil, o.logger.Error(
			"openai",
			fmt.Errorf("no embeddings returned from OpenAI"),
		)
	}

	// Convert from float64 to float32
	embedding := make([]float32, len(response.Data[0].Embedding))
	for i, v := range response.Data[0].Embedding {
		embedding[i] = float32(v)
	}

	return embedding, nil
}

//------------------------------------------------------------------------------
// Direct API Implementation
//------------------------------------------------------------------------------

// OpenAIEmbeddingProvider implements the EmbeddingProvider interface
// using direct calls to the OpenAI API (no SDK dependency)
type OpenAIEmbeddingProvider struct {
	logger *output.Logger
	// apiKey is the authentication key for the OpenAI API
	apiKey string
	// model is the specific embedding model to use
	model string
}

// NewOpenAIEmbeddingProvider creates a new OpenAI embedding provider
// using direct API calls
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
		logger: output.NewLogger(),
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
		return nil, o.logger.Error(
			"openai",
			fmt.Errorf("failed to marshal request: %w", err),
		)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, o.logger.Error(
			"openai",
			fmt.Errorf("failed to create HTTP request: %w", err),
		)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	// Execute the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, o.logger.Error(
			"openai",
			fmt.Errorf("failed to execute HTTP request: %w", err),
		)
	}
	defer resp.Body.Close()

	// Parse the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, o.logger.Error(
			"openai",
			fmt.Errorf("failed to read response body: %w", err),
		)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, o.logger.Error(
			"openai",
			fmt.Errorf("API error: %s, status code: %d", string(body), resp.StatusCode),
		)
	}

	var response EmbeddingResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, o.logger.Error(
			"openai",
			fmt.Errorf("failed to unmarshal response: %w", err),
		)
	}

	if len(response.Data) == 0 {
		return nil, o.logger.Error(
			"openai",
			fmt.Errorf("no embedding data returned"),
		)
	}

	return response.Data[0].Embedding, nil
}
