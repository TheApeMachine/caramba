//go:build amd64

package math

func signKernel(out, input []float64) {
	width := 2
	vectorSign := signVecSSE2

	if useAVX2 {
		width = 4
		vectorSign = signVecAVX2
	}

	limit := len(input) / width * width

	if limit > 0 {
		vectorSign(out[:limit], input[:limit])
	}

	for index := limit; index < len(input); index++ {
		switch {
		case input[index] > 0:
			out[index] = 1
		case input[index] < 0:
			out[index] = -1
		default:
			out[index] = 0
		}
	}
}
