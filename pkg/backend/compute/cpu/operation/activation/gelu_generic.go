//go:build !amd64 && !arm64

package activation

import "math"

func geluKernel(out, input []float64) {
	for index, value := range input {
		cube := value * value * value
		z := 0.7978845608028654 * (value + 0.044715*cube)

		out[index] = 0.5 * value * (1 + math.Tanh(z))
	}
}
