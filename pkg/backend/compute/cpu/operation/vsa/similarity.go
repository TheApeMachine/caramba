package vsa

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Similarity computes the dot-product cosine similarity between two VSA hypervectors.
Assumes both vectors are already L2-normalised (which Bundle guarantees), so the
dot product equals the cosine similarity directly.
shape=[N], data[0]=a, data[1]=b → out=[dot_product].
*/
type Similarity struct{}

/*
NewSimilarity instantiates a new Similarity operation.
*/
func NewSimilarity() *Similarity { return &Similarity{} }

/*
Forward returns a length-1 slice containing the dot product of data[0] and data[1].
*/
func (similarity *Similarity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("vsa.similarity", 2); err != nil {
		return nil, err
	}

	na, nb := len(stateDict.Inputs[0]), len(stateDict.Inputs[1])

	if na != nb {
		return nil, fmt.Errorf(
			"vsa.similarity: input lengths must match, got %d and %d",
			na, nb,
		)
	}

	if na == 0 {
		return nil, fmt.Errorf("vsa.similarity: empty vectors are not allowed")
	}

	stateDict.EnsureOperationOutLen(1)
	stateDict.Out[0] = similarityKernel(stateDict.Inputs[0], stateDict.Inputs[1])

	return stateDict, nil
}
