//go:build arm64

package math

import gomath "math"

//go:noescape
func sinVecNEON(dst, src []float64)

func sinKernel(out, input []float64) {
	limit := len(input) / 2 * 2

	if limit > 0 {
		sinVecNEON(out[:limit], input[:limit])
	}

	for index := limit; index < len(input); index++ {
		out[index] = gomath.Sin(input[index])
	}
}
