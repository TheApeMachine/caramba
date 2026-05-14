//go:build !amd64 && !arm64

package projection

func projectionMatmulGeneric(dst, input, weight []float64, M, K, N int) {
	for rowIndex := range M {
		for columnIndex := range N {
			sum := 0.0

			for innerIndex := range K {
				sum += input[rowIndex*K+innerIndex] * weight[innerIndex*N+columnIndex]
			}

			dst[rowIndex*N+columnIndex] = sum
		}
	}
}
