package math

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Matmul performs matrix multiplication A [M*K] x B [K*N] -> C [M*N].
shape = [M, K, N].
*/
type Matmul struct{}

func NewMatmul() *Matmul { return &Matmul{} }

func (matmul *Matmul) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("math.matmul", 2); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 3 {
		return nil, fmt.Errorf("math.matmul: len(shape)=%d, need >= 3", len(shape))
	}

	M, K, N := shape[0], shape[1], shape[2]

	mk := int64(M) * int64(K)
	kn := int64(K) * int64(N)
	mn := int64(M) * int64(N)

	if mk < 0 || mk > int64(math.MaxInt) {
		return nil, fmt.Errorf("math.matmul: M*K overflows int (M=%d K=%d)", M, K)
	}

	if kn < 0 || kn > int64(math.MaxInt) {
		return nil, fmt.Errorf("math.matmul: K*N overflows int (K=%d N=%d)", K, N)
	}

	if mn < 0 || mn > int64(math.MaxInt) {
		return nil, fmt.Errorf("math.matmul: M*N overflows int (M=%d N=%d)", M, N)
	}

	if len(stateDict.Inputs[0]) != int(mk) {
		return nil, fmt.Errorf(
			"math.matmul: len(input[0])=%d, need M*K=%d (M=%d K=%d)",
			len(stateDict.Inputs[0]), int(mk), M, K,
		)
	}

	if len(stateDict.Inputs[1]) != int(kn) {
		return nil, fmt.Errorf(
			"math.matmul: len(input[1])=%d, need K*N=%d (K=%d N=%d)",
			len(stateDict.Inputs[1]), int(kn), K, N,
		)
	}

	stateDict.EnsureOperationOutLen(int(mn))
	matmulKernel(stateDict.Out, stateDict.Inputs[0], stateDict.Inputs[1], M, K, N)

	return stateDict, nil
}
