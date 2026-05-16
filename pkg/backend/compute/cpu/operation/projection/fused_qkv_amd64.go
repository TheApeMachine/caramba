//go:build amd64

package projection

//go:noescape
func fusedQKVMatmulAVX2(dst, input, weight []float64, M, K, N int)

//go:noescape
func fusedQKVMatmulSSE2(dst, input, weight []float64, M, K, N int)

func fusedQKVKernel(dst, input, weight, bias []float64, M, K, N int) {
	if useAVX2 && useFMA {
		fusedQKVMatmulAVX2(dst, input, weight, M, K, N)
	} else {
		fusedQKVMatmulSSE2(dst, input, weight, M, K, N)
	}

	if len(bias) != 0 {
		addBias(dst, bias, M, N)
	}
}
