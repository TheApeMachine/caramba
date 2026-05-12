//go:build amd64

package attention

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA  bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA  = cpu.X86.HasFMA
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

func dotProduct(a, b []float64) float64 {
	// dotProductAVX2 uses VFMADD231PD; FMA required alongside AVX2.
	if useAVX2 && useFMA {
		return dotProductAVX2(a, b)
	}
	return dotProductSSE2(a, b)
}

func scaledAdd(dst, src []float64, scale float64) {
	// scaledAddAVX2 uses VFMADD231PD; FMA required alongside AVX2.
	if useAVX2 && useFMA {
		scaledAddAVX2(dst, src, scale)
	} else {
		scaledAddSSE2(dst, src, scale)
	}
}

func reduceMax(a []float64) float64 {
	if useAVX2 {
		return reduceMaxAVX2(a)
	}
	return reduceMaxSSE2(a)
}

func reduceSum(a []float64) float64 {
	if useAVX2 {
		return reduceSumAVX2(a)
	}
	return reduceSumSSE2(a)
}

func divScalar(dst []float64, s float64) {
	if useAVX2 {
		divScalarAVX2(dst, s)
	} else {
		divScalarSSE2(dst, s)
	}
}

//go:noescape
func attentionRowScoresAVX2(scores, q, K []float64, seqLen, headDim int, scale float64)

//go:noescape
func attentionRowScoresSSE2(scores, q, K []float64, seqLen, headDim int, scale float64)

//go:noescape
func attentionRowOutputAVX2(out, scores, V []float64, seqLen, headDim int)

//go:noescape
func attentionRowOutputSSE2(out, scores, V []float64, seqLen, headDim int)

func attentionRowScoresKernel(scores, q, K []float64, seqLen, headDim int, scale float64) {
	if useAVX2 && useFMA {
		attentionRowScoresAVX2(scores, q, K, seqLen, headDim, scale)
		return
	}

	attentionRowScoresSSE2(scores, q, K, seqLen, headDim, scale)
}

func attentionRowOutputKernel(out, scores, V []float64, seqLen, headDim int) {
	if useAVX2 && useFMA {
		attentionRowOutputAVX2(out, scores, V, seqLen, headDim)
		return
	}

	attentionRowOutputSSE2(out, scores, V, seqLen, headDim)
}
