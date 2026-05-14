//go:build amd64

package math

func mulKernel(out, left, right []float64) {
	width := 2
	vectorMul := mulVecSSE2

	if useAVX2 {
		width = 4
		vectorMul = mulVecAVX2
	}

	limit := len(left) / width * width

	if limit > 0 {
		vectorMul(out[:limit], left[:limit], right[:limit])
	}

	for index := limit; index < len(left); index++ {
		out[index] = left[index] * right[index]
	}
}
