package projection

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
TiedEmbedding projects hidden states back to vocabulary logits using the
transposed token embedding matrix supplied by the state dict.
*/
type TiedEmbedding struct{}

/*
NewTiedEmbedding instantiates a stateless tied embedding projection.
*/
func NewTiedEmbedding(weight []float64, vocabSize, dModel int) (*TiedEmbedding, error) {
	return &TiedEmbedding{}, nil
}

/*
Forward computes logits = x @ weight.
*/
func (tiedEmbedding *TiedEmbedding) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("projection.tied_embedding"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("projection.tied_embedding: shape is required")
	}

	K := stateDict.DModel

	if K == 0 {
		K = shape[len(shape)-1]
	}

	N := stateDict.VocabSize

	if K <= 0 {
		return nil, fmt.Errorf("projection.tied_embedding: d_model must be positive, got %d", K)
	}

	if N <= 0 {
		return nil, fmt.Errorf("projection.tied_embedding: vocab_size must be positive, got %d", N)
	}

	if shape[len(shape)-1] != K {
		return nil, fmt.Errorf(
			"projection.tied_embedding: shape last dim %d does not match DModel=%d",
			shape[len(shape)-1], K,
		)
	}

	M := 1

	for dimensionIndex := 0; dimensionIndex < len(shape)-1; dimensionIndex++ {
		dimension := shape[dimensionIndex]

		if dimension < 0 {
			return nil, fmt.Errorf(
				"projection.tied_embedding: shape[%d]=%d must be non-negative",
				dimensionIndex, dimension,
			)
		}

		if dimension == 0 {
			M = 0

			break
		}

		if M > math.MaxInt/dimension {
			return nil, fmt.Errorf("projection.tied_embedding: batch product overflows int")
		}

		M *= dimension
	}

	if M > len(stateDict.Inputs[0])/K {
		return nil, fmt.Errorf(
			"projection.tied_embedding: input length %d is insufficient for M=%d and K=%d",
			len(stateDict.Inputs[0]), M, K,
		)
	}

	if int64(K)*int64(N) < 0 || int64(K)*int64(N) > int64(math.MaxInt) {
		return nil, fmt.Errorf("projection.tied_embedding: K*N overflows int")
	}

	if len(stateDict.Weight) != K*N {
		return nil, fmt.Errorf(
			"projection.tied_embedding: weight length %d does not match K*N=%d",
			len(stateDict.Weight), K*N,
		)
	}

	if int64(M)*int64(N) < 0 || int64(M)*int64(N) > int64(math.MaxInt) {
		return nil, fmt.Errorf("projection.tied_embedding: M*N overflows int")
	}

	stateDict.EnsureOperationOutLen(M * N)
	tiedEmbeddingKernel(stateDict.Out, stateDict.Inputs[0], stateDict.Weight, M, K, N)

	return stateDict, nil
}
