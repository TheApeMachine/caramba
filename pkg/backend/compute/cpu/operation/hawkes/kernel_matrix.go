package hawkes

import (
	"fmt"
	stdmath "math"
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

func (op *KernelMatrix) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 {
		panic(fmt.Errorf("hawkes: KernelMatrix: len(shape)=%d, need >= 1", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("hawkes: KernelMatrix: len(data)=%d, need 3", len(data)).Error())
	}

	T := shape[0]
	times := data[0]
	alpha := data[1][0]
	beta := data[2][0]

	if len(times) != T {
		panic(fmt.Errorf(
			"hawkes: KernelMatrix: len(times)=%d, need T=%d",
			len(times), T,
		).Error())
	}

	out := make([]float64, T*T)
	applyKernelMatrix(out, times, alpha, beta, T)

	return out
}

func applyKernelMatrixScalar(out, times []float64, alpha, beta float64, T int) {
	for row := range T {
		ti := times[row]

		for col := row + 1; col < T; col++ {
			dt := times[col] - ti
			out[row*T+col] = alpha * stdmath.Exp(-beta*dt)
		}
	}
}
