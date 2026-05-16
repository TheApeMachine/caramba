//go:build arm64

package projection

//go:noescape
func linearMatmulNEON(dst, input, weight []float64, M, K, N int)

func linearKernel(dst, input, weight, bias []float64, M, K, N int) {
	linearMatmulNEON(dst, input, weight, M, K, N)

	if len(bias) != 0 {
		addBias(dst, bias, M, N)
	}
}
