//go:build amd64

package math

import gomath "math"

//go:noescape
func sinVecAVX2(dst, src []float64)

//go:noescape
func sinVecSSE2(dst, src []float64)

func sinKernel(out, input []float64) {
	width := 2
	vectorSin := sinVecSSE2

	if useAVX2 && useFMA {
		width = 4
		vectorSin = sinVecAVX2
	}

	limit := len(input) / width * width

	if limit > 0 {
		vectorSin(out[:limit], input[:limit])
	}

	for index := limit; index < len(input); index++ {
		out[index] = gomath.Sin(input[index])
	}
}
