//go:build arm64

package projection

//go:noescape
func fusedQKVMatmulNEON(dst, input, weight []float64, M, K, N int)

func fusedQKVKernel(dst, input, weight, bias []float64, M, K, N int) {
	fusedQKVMatmulNEON(dst, input, weight, M, K, N)

	if len(bias) != 0 {
		addBias(dst, bias, M, N)
	}
}
