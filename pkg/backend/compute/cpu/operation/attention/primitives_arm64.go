//go:build arm64

package attention

import (
	"fmt"
	"math"
)

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

// dotProduct returns Σ a[i]*b[i]. Panics on length mismatch.
func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("attention: dotProduct: length mismatch len(a)=%d len(b)=%d", len(a), len(b)))
	}

	if len(a) == 0 {
		return 0
	}

	return dotProductNEON(a, b)
}

// scaledAdd performs dst[i] += scale * src[i]. Panics on length mismatch.
func scaledAdd(dst, src []float64, scale float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("attention: scaledAdd: length mismatch len(dst)=%d len(src)=%d", len(dst), len(src)))
	}

	if len(dst) == 0 {
		return
	}

	scaledAddNEON(dst, src, scale)
}

// reduceMax returns the maximum element, or -Inf for an empty slice.
func reduceMax(a []float64) float64 {
	if len(a) == 0 {
		return math.Inf(-1)
	}

	return reduceMaxNEON(a)
}

// reduceSum returns the sum of all elements, or 0 for an empty slice.
func reduceSum(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}

	return reduceSumNEON(a)
}

// divScalar divides each element by s in place. No-op for empty dst;
// panics if s == 0 to surface intent rather than emit ±Inf/NaN silently.
func divScalar(dst []float64, s float64) {
	if len(dst) == 0 {
		return
	}

	if s == 0 {
		panic("attention: divScalar: divisor is zero")
	}

	divScalarNEON(dst, s)
}

// attentionRowScoresKernel writes scores[j] = scale * dot(q, K[j*headDim:]).
// Validates that scores, q, K can be safely indexed by seqLen and headDim.
func attentionRowScoresKernel(scores, q, K []float64, seqLen, headDim int, scale float64) {
	if seqLen < 0 || headDim < 0 {
		panic(fmt.Sprintf("attention: attentionRowScoresKernel: negative dims seqLen=%d headDim=%d", seqLen, headDim))
	}

	if len(scores) < seqLen {
		panic(fmt.Sprintf("attention: attentionRowScoresKernel: len(scores)=%d < seqLen=%d", len(scores), seqLen))
	}

	if len(q) < headDim {
		panic(fmt.Sprintf("attention: attentionRowScoresKernel: len(q)=%d < headDim=%d", len(q), headDim))
	}

	if len(K) < seqLen*headDim {
		panic(fmt.Sprintf("attention: attentionRowScoresKernel: len(K)=%d < seqLen*headDim=%d", len(K), seqLen*headDim))
	}

	attentionRowScoresNEON(scores, q, K, seqLen, headDim, scale)
}

// attentionRowOutputKernel writes out[d] = Σ_j scores[j] * V[j*headDim + d].
// Validates out/scores/V lengths.
func attentionRowOutputKernel(out, scores, V []float64, seqLen, headDim int) {
	if seqLen < 0 || headDim < 0 {
		panic(fmt.Sprintf("attention: attentionRowOutputKernel: negative dims seqLen=%d headDim=%d", seqLen, headDim))
	}

	if len(out) < headDim {
		panic(fmt.Sprintf("attention: attentionRowOutputKernel: len(out)=%d < headDim=%d", len(out), headDim))
	}

	if len(scores) < seqLen {
		panic(fmt.Sprintf("attention: attentionRowOutputKernel: len(scores)=%d < seqLen=%d", len(scores), seqLen))
	}

	if len(V) < seqLen*headDim {
		panic(fmt.Sprintf("attention: attentionRowOutputKernel: len(V)=%d < seqLen*headDim=%d", len(V), seqLen*headDim))
	}

	attentionRowOutputNEON(out, scores, V, seqLen, headDim)
}
