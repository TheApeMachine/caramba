//go:build !amd64 && !arm64

package math

func signKernel(out, input []float64) {
	for index, value := range input {
		switch {
		case value > 0:
			out[index] = 1
		case value < 0:
			out[index] = -1
		default:
			out[index] = 0
		}
	}
}
