//go:build arm64

package convolution

//go:noescape
func dotProductNEON(a, b []float64) float64

//go:noescape
func scaledAddNEON(dst, src []float64, scale float64)

func dotProduct(a, b []float64) float64 {
	return dotProductNEON(a, b)
}

func scaledAdd(dst, src []float64, scale float64) {
	scaledAddNEON(dst, src, scale)
}
