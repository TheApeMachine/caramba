//go:build arm64

package attention

//go:noescape
func dotProductNEON(a, b []float64) float64

//go:noescape
func scaledAddNEON(dst, src []float64, scale float64)

//go:noescape
func reduceMaxNEON(a []float64) float64

//go:noescape
func reduceSumNEON(a []float64) float64

//go:noescape
func divScalarNEON(dst []float64, s float64)

//go:noescape
func attentionRowScoresNEON(scores, q, K []float64, seqLen, headDim int, scale float64)

//go:noescape
func attentionRowOutputNEON(out, scores, V []float64, seqLen, headDim int)

func dotProduct(a, b []float64) float64 {
	return dotProductNEON(a, b)
}

func scaledAdd(dst, src []float64, scale float64) {
	scaledAddNEON(dst, src, scale)
}

func reduceMax(a []float64) float64 {
	return reduceMaxNEON(a)
}

func reduceSum(a []float64) float64 {
	return reduceSumNEON(a)
}

func divScalar(dst []float64, s float64) {
	divScalarNEON(dst, s)
}

func attentionRowScoresKernel(scores, q, K []float64, seqLen, headDim int, scale float64) {
	attentionRowScoresNEON(scores, q, K, seqLen, headDim, scale)
}

func attentionRowOutputKernel(out, scores, V []float64, seqLen, headDim int) {
	attentionRowOutputNEON(out, scores, V, seqLen, headDim)
}
