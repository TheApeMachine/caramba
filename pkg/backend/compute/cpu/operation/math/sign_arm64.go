//go:build arm64

package math

func signKernel(out, input []float64) {
	signVecNEON(out, input)

	if len(input)%2 != 0 {
		index := len(input) - 1

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
