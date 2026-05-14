//go:build amd64

package math

func expKernel(out, input []float64) {
	width := 2
	vectorExp := expVecSSE2

	if useAVX2 {
		width = 4
		vectorExp = expVecAVX2
	}

	limit := len(input) / width * width

	if limit > 0 {
		vectorExp(out[:limit], input[:limit])
	}

	if limit < len(input) {
		scalarExpTailKernel(out, input, limit)
	}
}
