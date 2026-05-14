//go:build amd64

package math

import "math/rand"

func dropoutKernel(out, input []float64, probability float64, training bool) {
	if !training || probability == 0 {
		copy(out, input)

		return
	}

	scale := 1.0 / (1.0 - probability)

	for index, value := range input {
		if rand.Float64() >= probability {
			out[index] = value * scale
		}
	}
}
