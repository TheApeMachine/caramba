//go:build !amd64 && !arm64

package vsa

import "math"

func bundleKernel(dst []float64, inputs [][]float64) {
	for _, input := range inputs {
		for index := range input {
			dst[index] += input[index]
		}
	}

	sumSquares := 0.0

	for _, value := range dst {
		sumSquares += value * value
	}

	norm := math.Sqrt(sumSquares)

	if norm <= l2NormEpsilon {
		return
	}

	scale := 1.0 / norm

	for index := range dst {
		dst[index] *= scale
	}
}
