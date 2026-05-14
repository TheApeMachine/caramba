//go:build !amd64 && !arm64

package math

func addKernel(out, left, right []float64) {
	for index := range left {
		out[index] = left[index] + right[index]
	}
}
