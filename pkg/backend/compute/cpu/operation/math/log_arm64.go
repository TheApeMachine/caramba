//go:build arm64

package math

func logKernel(out, input []float64) {
	logVecNEON(out, input)
}
