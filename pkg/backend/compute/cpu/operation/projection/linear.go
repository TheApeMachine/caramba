package projection

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Linear applies a learnable affine transformation: output = x @ weight + bias.

The state dict supplies:
  - OpShape: [M, InFeatures]
  - Inputs[0]: x, flattened [M * InFeatures]
  - Weight: pre-transposed [InFeatures * OutFeatures]
  - Bias: optional [OutFeatures]
  - InFeatures, OutFeatures
*/
type Linear struct{}

/*
NewLinear instantiates a stateless Linear operation.
*/
func NewLinear(args ...int) *Linear {
	return &Linear{}
}

/*
Forward computes output = x @ weight + bias.
*/
func (linear *Linear) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("projection.linear"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("projection.linear: shape is required")
	}

	K := stateDict.InFeatures
	N := stateDict.OutFeatures

	if len(shape) > 1 && K == 0 {
		K = shape[len(shape)-1]
	}

	if K <= 0 {
		return nil, fmt.Errorf("projection.linear: in_features must be positive, got %d", K)
	}

	if N <= 0 {
		return nil, fmt.Errorf("projection.linear: out_features must be positive, got %d", N)
	}

	if len(stateDict.Inputs[0])%K != 0 {
		return nil, fmt.Errorf(
			"projection.linear: input length %d is not divisible by K=%d",
			len(stateDict.Inputs[0]), K,
		)
	}

	M := len(stateDict.Inputs[0]) / K

	if int64(K)*int64(N) < 0 || int64(K)*int64(N) > int64(math.MaxInt) {
		return nil, fmt.Errorf("projection.linear: K*N overflows int")
	}

	if len(stateDict.Weight) != K*N {
		return nil, fmt.Errorf(
			"projection.linear: weight length %d does not match K*N=%d",
			len(stateDict.Weight), K*N,
		)
	}

	if len(stateDict.Bias) != 0 && len(stateDict.Bias) != N {
		return nil, fmt.Errorf(
			"projection.linear: bias length %d does not match N=%d",
			len(stateDict.Bias), N,
		)
	}

	if int64(M)*int64(N) < 0 || int64(M)*int64(N) > int64(math.MaxInt) {
		return nil, fmt.Errorf("projection.linear: M*N overflows int")
	}

	stateDict.EnsureOperationOutLen(M * N)
	linearKernel(
		stateDict.Out, stateDict.Inputs[0], stateDict.Weight, stateDict.Bias,
		M, K, N,
	)

	return stateDict, nil
}

func transposeF64(src []float64, rows, cols int) []float64 {
	dst := make([]float64, rows*cols)

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			dst[col*rows+row] = src[row*cols+col]
		}
	}

	return dst
}

func addBias(out, bias []float64, M, N int) {
	for rowIndex := 0; rowIndex < M; rowIndex++ {
		row := out[rowIndex*N : rowIndex*N+N]

		for columnIndex, value := range bias {
			row[columnIndex] += value
		}
	}
}
