//go:build arm64

package math

func addKernel(out, left, right []float64) {
	addVecNEON(out, left, right)

	if len(left)%2 != 0 {
		index := len(left) - 1
		out[index] = left[index] + right[index]
	}
}
