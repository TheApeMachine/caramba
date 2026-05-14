//go:build arm64

package math

func mulKernel(out, left, right []float64) {
	mulVecNEON(out, left, right)

	if len(left)%2 != 0 {
		index := len(left) - 1
		out[index] = left[index] * right[index]
	}
}
