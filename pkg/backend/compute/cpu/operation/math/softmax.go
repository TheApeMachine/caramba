package math

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Softmax computes softmax over the last dimension of the input.
Each row passes through a dedicated AVX2/SSE2/NEON kernel that fuses max,
exp (polynomial range-reduced), sum, and reciprocal-divide inline.
*/
type Softmax struct{}

func NewSoftmax() *Softmax { return &Softmax{} }

func (softmax *Softmax) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.softmax"); err != nil {
		return nil, err
	}

	dimSize := stateDict.OperationLastDim()

	if dimSize <= 0 {
		return nil, fmt.Errorf("math.softmax: last dimension must be positive, got %d", dimSize)
	}

	if len(stateDict.Inputs[0])%dimSize != 0 {
		return nil, fmt.Errorf(
			"math.softmax: input length %d is not divisible by dim %d",
			len(stateDict.Inputs[0]), dimSize,
		)
	}

	softmaxKernel(stateDict.Out, stateDict.Inputs[0], dimSize)

	return stateDict, nil
}
