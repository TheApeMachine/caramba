//go:build amd64

package math

import gomath "math"

//go:noescape
func cosVecAVX2(dst, src []float64)

//go:noescape
func cosVecSSE2(dst, src []float64)

func cosKernel(out, input []float64) {
	width := 2
	vectorCos := cosVecSSE2

	if useAVX2 && useFMA {
		width = 4
		vectorCos = cosVecAVX2
	}

	limit := len(input) / width * width

	if limit > 0 {
		vectorCos(out[:limit], input[:limit])
	}

	for index := limit; index < len(input); index++ {
		out[index] = gomath.Cos(input[index])
	}
}
