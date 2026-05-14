/*
Package embedding implements token-embedding lookup for transformer models.
*/
package embedding

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
TokenEmbedding maps token IDs to dense embedding vectors supplied by the state dict.
*/
type TokenEmbedding struct{}

/*
NewTokenEmbedding instantiates a stateless token embedding operation.
*/
func NewTokenEmbedding(vocabSize, dModel int, initStd float64) *TokenEmbedding {
	return &TokenEmbedding{}
}

/*
Forward looks up the embedding vector for each token ID in Inputs[0].
*/
func (tokenEmbedding *TokenEmbedding) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("embedding.token_embedding"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("embedding.token_embedding: shape is required")
	}

	vocabSize := stateDict.VocabSize
	dModel := stateDict.DModel

	if vocabSize <= 0 {
		return nil, fmt.Errorf("embedding.token_embedding: vocab_size must be positive, got %d", vocabSize)
	}

	if dModel <= 0 {
		return nil, fmt.Errorf("embedding.token_embedding: d_model must be positive, got %d", dModel)
	}

	if len(stateDict.Weight) != vocabSize*dModel {
		return nil, fmt.Errorf(
			"embedding.token_embedding: weight length %d does not match vocab_size*d_model=%d",
			len(stateDict.Weight), vocabSize*dModel,
		)
	}

	tokenCount, err := tokenEmbeddingShapeSize(shape)

	if err != nil {
		return nil, err
	}

	tokens := stateDict.Inputs[0]

	if len(tokens) != tokenCount {
		return nil, fmt.Errorf(
			"embedding.token_embedding: token length %d does not match shape product %d",
			len(tokens), tokenCount,
		)
	}

	for index, token := range tokens {
		tokenID := int(token)

		if token != math.Trunc(token) || tokenID < 0 || tokenID >= vocabSize {
			return nil, fmt.Errorf(
				"embedding.token_embedding: token[%d]=%v is outside vocab range [0,%d)",
				index, token, vocabSize,
			)
		}
	}

	stateDict.EnsureOperationOutLen(tokenCount * dModel)
	tokenEmbeddingKernel(stateDict.Out, tokens, stateDict.Weight, dModel)

	return stateDict, nil
}

func tokenEmbeddingShapeSize(shape []int) (int, error) {
	size := 1

	for index, dimension := range shape {
		if dimension < 0 {
			return 0, fmt.Errorf(
				"embedding.token_embedding: shape[%d]=%d must be non-negative",
				index, dimension,
			)
		}

		if dimension == 0 {
			return 0, nil
		}

		if size > math.MaxInt/dimension {
			return 0, fmt.Errorf("embedding.token_embedding: shape product overflows int")
		}

		size *= dimension
	}

	return size, nil
}
