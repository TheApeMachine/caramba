//go:build amd64

package attention

import (
	"fmt"
	"math"

	"golang.org/x/sys/cpu"
)

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func dotProductAVX2(a, b []float64) float64

//go:noescape
func dotProductSSE2(a, b []float64) float64

//go:noescape
func scaledAddAVX2(dst, src []float64, scale float64)

//go:noescape
func scaledAddSSE2(dst, src []float64, scale float64)

//go:noescape
func reduceMaxAVX2(a []float64) float64

//go:noescape
func reduceMaxSSE2(a []float64) float64

//go:noescape
func reduceSumAVX2(a []float64) float64

//go:noescape
func reduceSumSSE2(a []float64) float64

//go:noescape
func divScalarAVX2(dst []float64, s float64)

//go:noescape
func divScalarSSE2(dst []float64, s float64)

//go:noescape
func attentionRowScoresAVX2(scores, q, K []float64, seqLen, headDim int, scale float64)

//go:noescape
func attentionRowScoresSSE2(scores, q, K []float64, seqLen, headDim int, scale float64)

//go:noescape
func attentionRowOutputAVX2(out, scores, V []float64, seqLen, headDim int)

//go:noescape
func attentionRowOutputSSE2(out, scores, V []float64, seqLen, headDim int)

// dotProduct returns Σ a[i]*b[i]. Panics on length mismatch.
func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("attention: dotProduct: length mismatch len(a)=%d len(b)=%d", len(a), len(b)))
	}

	if len(a) == 0 {
		return 0
	}

	if useAVX2 && useFMA {
		return dotProductAVX2(a, b)
	}

	return dotProductSSE2(a, b)
}

// scaledAdd performs dst[i] += scale*src[i]. Panics on length mismatch.
func scaledAdd(dst, src []float64, scale float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("attention: scaledAdd: length mismatch len(dst)=%d len(src)=%d", len(dst), len(src)))
	}

	if len(dst) == 0 {
		return
	}

	if useAVX2 && useFMA {
		scaledAddAVX2(dst, src, scale)
		return
	}

	scaledAddSSE2(dst, src, scale)
}

// reduceMax returns the maximum element, or -Inf for an empty slice.
func reduceMax(a []float64) float64 {
	if len(a) == 0 {
		return math.Inf(-1)
	}

	if useAVX2 {
		return reduceMaxAVX2(a)
	}

	return reduceMaxSSE2(a)
}

// reduceSum returns the sum of all elements, or 0 for an empty slice.
func reduceSum(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}

	if useAVX2 {
		return reduceSumAVX2(a)
	}

	return reduceSumSSE2(a)
}

// divScalar divides each element by s in place. Panics if s == 0.
func divScalar(dst []float64, s float64) {
	if len(dst) == 0 {
		return
	}

	if s == 0 {
		panic("attention: divScalar: divisor is zero")
	}

	if useAVX2 {
		divScalarAVX2(dst, s)
		return
	}

	divScalarSSE2(dst, s)
}

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

	if useAVX2 && useFMA {
		attentionRowScoresAVX2(scores, q, K, seqLen, headDim, scale)
		return
	}

	attentionRowScoresSSE2(scores, q, K, seqLen, headDim, scale)
}

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

	if useAVX2 && useFMA {
		attentionRowOutputAVX2(out, scores, V, seqLen, headDim)
		return
	}

	attentionRowOutputSSE2(out, scores, V, seqLen, headDim)
}
