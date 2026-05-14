//go:build arm64

package math

func expKernel(out, input []float64) {
	expVecNEON(out, input)
}
