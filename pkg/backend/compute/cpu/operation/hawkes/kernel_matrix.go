package hawkes

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
KernelMatrix builds the excitation kernel matrix for a single-process Hawkes:

	K[i,j] = alpha * exp(-beta * (t_j - t_i))  for j > i
	K[i,j] = 0                                  otherwise

shape = [T, alpha_idx, beta_idx]   (alpha_idx, beta_idx unused; kept for shape consistency)
data[0] = times [T]
data[1] = alpha [1]
data[2] = beta  [1]

Returns matrix [T*T] row-major.
*/
type KernelMatrix struct{}

func NewKernelMatrix() *KernelMatrix { return &KernelMatrix{} }

func (kernelMatrix *KernelMatrix) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("hawkes.kernel_matrix: len(shape)=%d, need >= 1", len(shape))
	}

	if err := stateDict.RequireOperationInputs("hawkes.kernel_matrix", 3); err != nil {
		return nil, err
	}

	eventCount := shape[0]

	if eventCount <= 0 {
		return nil, fmt.Errorf("hawkes.kernel_matrix: T=%d, need T > 0", eventCount)
	}

	times := stateDict.Inputs[0]

	if len(stateDict.Inputs[1]) < 1 || len(stateDict.Inputs[2]) < 1 {
		return nil, fmt.Errorf("hawkes.kernel_matrix: alpha and beta must be non-empty")
	}

	alpha := stateDict.Inputs[1][0]
	beta := stateDict.Inputs[2][0]

	if len(times) != eventCount {
		return nil, fmt.Errorf(
			"hawkes.kernel_matrix: len(times)=%d, need T=%d",
			len(times), eventCount,
		)
	}

	for index := 0; index+1 < eventCount; index++ {
		if times[index] >= times[index+1] {
			return nil, fmt.Errorf(
				"hawkes.kernel_matrix: times must be strictly increasing (indices %d,%d: %v >= %v)",
				index, index+1, times[index], times[index+1],
			)
		}
	}

	stateDict.EnsureOperationOutLen(eventCount * eventCount)
	applyKernelMatrix(stateDict.Out, times, alpha, beta, eventCount)

	return stateDict, nil
}

func applyKernelMatrixScalar(out, times []float64, alpha, beta float64, T int) {
	for row := range T {
		ti := times[row]

		for col := row + 1; col < T; col++ {
			dt := times[col] - ti
			out[row*T+col] = alpha * math.Exp(-beta*dt)
		}
	}
}
