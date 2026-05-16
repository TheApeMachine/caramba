//go:build amd64

package projection

//go:noescape
func linearMatmulAVX2(dst, input, weight []float64, M, K, N int)

//go:noescape
func linearMatmulSSE2(dst, input, weight []float64, M, K, N int)

func linearKernel(dst, input, weight, bias []float64, M, K, N int) {
	if useAVX2 && useFMA {
		linearMatmulAVX2(dst, input, weight, M, K, N)
	} else {
		linearMatmulSSE2(dst, input, weight, M, K, N)
	}

	if len(bias) != 0 {
		addBias(dst, bias, M, N)
	}
}
