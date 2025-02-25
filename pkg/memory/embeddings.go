package memory

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/theapemachine/errnie"
)

// EmbeddingsProvider defines the interface for generating text embeddings
type EmbeddingsProvider interface {
	// GetEmbeddings generates vector embeddings for the given text
	GetEmbeddings(ctx context.Context, text string) ([]float32, error)
}

// OpenAIEmbeddings implements the EmbeddingsProvider interface using OpenAI
type OpenAIEmbeddings struct {
	client           *openai.Client
	embeddingService *openai.EmbeddingService
	model            openai.EmbeddingModel
}

// NewOpenAIEmbeddings creates a new OpenAIEmbeddings provider
func NewOpenAIEmbeddings(apiKey string, model string) *OpenAIEmbeddings {
	if model == "" {
		model = openai.EmbeddingModelTextEmbedding3Small // Default to modern embedding model
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	return &OpenAIEmbeddings{
		client:           client,
		embeddingService: openai.NewEmbeddingService(),
		model:            model,
	}
}

// GetEmbeddings implements the EmbeddingsProvider interface for OpenAI
func (o *OpenAIEmbeddings) GetEmbeddings(ctx context.Context, text string) ([]float32, error) {
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
		errnie.Error(fmt.Errorf("failed to get OpenAI embeddings: %w", err))
		return nil, err
	}

	if len(response.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned from OpenAI")
	}

	// Convert from float64 to float32
	embedding := make([]float32, len(response.Data[0].Embedding))
	for i, v := range response.Data[0].Embedding {
		embedding[i] = float32(v)
	}

	return embedding, nil
}
