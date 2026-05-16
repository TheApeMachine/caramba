package math

import (
	"fmt"
	stdmath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
MatmulAdd performs matrix multiplication with a broadcast or full output bias.
shape = [M, K, N].
*/
type MatmulAdd struct{}

func NewMatmulAdd() *MatmulAdd {
	return &MatmulAdd{}
}

func (matmulAdd *MatmulAdd) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("math.matmul_add", 3); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 3 {
		return nil, fmt.Errorf("math.matmul_add: len(shape)=%d, need >= 3", len(shape))
	}

	leftRows, sharedDim, rightCols := shape[0], shape[1], shape[2]
	leftLen, rightLen, outputLen, err := matmulLengths(leftRows, sharedDim, rightCols)

	if err != nil {
		return nil, err
	}

	if len(stateDict.Inputs[0]) != leftLen {
		return nil, fmt.Errorf(
			"math.matmul_add: len(input[0])=%d, need M*K=%d",
			len(stateDict.Inputs[0]), leftLen,
		)
	}

	if len(stateDict.Inputs[1]) != rightLen {
		return nil, fmt.Errorf(
			"math.matmul_add: len(input[1])=%d, need K*N=%d",
			len(stateDict.Inputs[1]), rightLen,
		)
	}

	if err := requireMatmulBias(len(stateDict.Inputs[2]), leftRows, rightCols); err != nil {
		return nil, err
	}

	stateDict.EnsureOperationOutLen(outputLen)
	matmulAddKernel(
		stateDict.Out,
		stateDict.Inputs[0],
		stateDict.Inputs[1],
		stateDict.Inputs[2],
		leftRows,
		sharedDim,
		rightCols,
	)

	return stateDict, nil
}

func matmulLengths(leftRows, sharedDim, rightCols int) (int, int, int, error) {
	leftLen := int64(leftRows) * int64(sharedDim)
	rightLen := int64(sharedDim) * int64(rightCols)
	outputLen := int64(leftRows) * int64(rightCols)

	if leftLen < 0 || leftLen > int64(stdmath.MaxInt) {
		return 0, 0, 0, fmt.Errorf("math.matmul_add: M*K overflows int")
	}

	if rightLen < 0 || rightLen > int64(stdmath.MaxInt) {
		return 0, 0, 0, fmt.Errorf("math.matmul_add: K*N overflows int")
	}

	if outputLen < 0 || outputLen > int64(stdmath.MaxInt) {
		return 0, 0, 0, fmt.Errorf("math.matmul_add: M*N overflows int")
	}

	return int(leftLen), int(rightLen), int(outputLen), nil
}

func requireMatmulBias(biasLen, leftRows, rightCols int) error {
	if biasLen == rightCols || biasLen == leftRows*rightCols {
		return nil
	}

	return fmt.Errorf(
		"math.matmul_add: bias length %d must be N=%d or M*N=%d",
		biasLen, rightCols, leftRows*rightCols,
	)
}

func matmulAddKernel(
	output, left, right, bias []float64,
	leftRows, sharedDim, rightCols int,
) {
	fullBias := len(bias) == leftRows*rightCols

	for row := range leftRows {
		for col := range rightCols {
			accumulator := bias[col]

			if fullBias {
				accumulator = bias[row*rightCols+col]
			}

			for shared := range sharedDim {
				accumulator += left[row*sharedDim+shared] * right[shared*rightCols+col]
			}

			output[row*rightCols+col] = accumulator
		}
	}
}
