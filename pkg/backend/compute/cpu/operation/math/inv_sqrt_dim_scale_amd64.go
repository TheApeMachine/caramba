//go:build amd64

package math

func invSqrtDimScaleKernel(out, input []float64, scale float64) {
	copy(out, input)

	width := 2
	vectorMulScalar := mulScalarSSE2

	if useAVX2 {
		width = 4
		vectorMulScalar = mulScalarAVX2
	}

	limit := len(out) / width * width

	if limit > 0 {
		vectorMulScalar(out[:limit], scale)
	}

	for index := limit; index < len(out); index++ {
		out[index] *= scale
	}
}
