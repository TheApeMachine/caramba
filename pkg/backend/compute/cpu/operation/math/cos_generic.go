//go:build !amd64 && !arm64

package math

import gomath "math"

func cosKernel(out, input []float64) {
	for index, value := range input {
		out[index] = gomath.Cos(value)
	}
}
