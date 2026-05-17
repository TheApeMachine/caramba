//go:build arm64

package math

import gomath "math"

//go:noescape
func cosVecNEON(dst, src []float64)

func cosKernel(out, input []float64) {
	limit := len(input) / 2 * 2

	if limit > 0 {
		cosVecNEON(out[:limit], input[:limit])
	}

	for index := limit; index < len(input); index++ {
		out[index] = gomath.Cos(input[index])
	}
}
