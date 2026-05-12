//go:build amd64

package math

import (
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
func reduceSumAVX2(a []float64) float64

//go:noescape
func reduceSumSSE2(a []float64) float64

//go:noescape
func reduceMaxAVX2(a []float64) float64

//go:noescape
func reduceMaxSSE2(a []float64) float64

//go:noescape
func divScalarAVX2(dst []float64, s float64)

//go:noescape
func divScalarSSE2(dst []float64, s float64)

//go:noescape
func addVecAVX2(dst, a, b []float64)

//go:noescape
func addVecSSE2(dst, a, b []float64)

//go:noescape
func mulVecAVX2(dst, a, b []float64)

//go:noescape
func mulVecSSE2(dst, a, b []float64)

//go:noescape
func mulScalarAVX2(dst []float64, s float64)

//go:noescape
func mulScalarSSE2(dst []float64, s float64)

//go:noescape
func reduceSumSqAVX2(a []float64) float64

//go:noescape
func reduceSumSqSSE2(a []float64) float64

func reduceSum(a []float64) float64 {
	limit := alignedLen(len(a))
	sum := 0.0

	if useAVX2 {
		if limit > 0 {
			sum = reduceSumAVX2(a[:limit])
		}
	} else if limit > 0 {
		sum = reduceSumSSE2(a[:limit])
	}

	for _, value := range a[limit:] {
		sum += value
	}

	return sum
}

// reduceMax returns the largest element of a.
// For an empty slice it returns -math.MaxFloat64 (identity for max reductions).
func reduceMax(a []float64) float64 {
	if len(a) == 0 {
		return -math.MaxFloat64
	}

	limit := alignedLen(len(a))
	start := limit
	var maxValue float64

	if useAVX2 {
		if limit > 0 {
			maxValue = reduceMaxAVX2(a[:limit])
		}
	} else if limit > 0 {
		maxValue = reduceMaxSSE2(a[:limit])
	}

	if limit == 0 {
		maxValue = a[0]
		start = 1
	}

	for _, value := range a[start:] {
		if value > maxValue {
			maxValue = value
		}
	}

	return maxValue
}

func divScalar(dst []float64, s float64) {
	limit := alignedLen(len(dst))

	if useAVX2 {
		if limit > 0 {
			divScalarAVX2(dst[:limit], s)
		}
	} else if limit > 0 {
		divScalarSSE2(dst[:limit], s)
	}

	for index := limit; index < len(dst); index++ {
		dst[index] /= s
	}
}

func addVec(dst, a, b []float64) {
	limit := alignedLen(len(a))

	if useAVX2 {
		if limit > 0 {
			addVecAVX2(dst[:limit], a[:limit], b[:limit])
		}
	} else if limit > 0 {
		addVecSSE2(dst[:limit], a[:limit], b[:limit])
	}

	for index := limit; index < len(a); index++ {
		dst[index] = a[index] + b[index]
	}
}

func mulVec(dst, a, b []float64) {
	limit := alignedLen(len(a))

	if useAVX2 {
		if limit > 0 {
			mulVecAVX2(dst[:limit], a[:limit], b[:limit])
		}
	} else if limit > 0 {
		mulVecSSE2(dst[:limit], a[:limit], b[:limit])
	}

	for index := limit; index < len(a); index++ {
		dst[index] = a[index] * b[index]
	}
}

func mulScalar(dst []float64, s float64) {
	limit := alignedLen(len(dst))

	if useAVX2 {
		if limit > 0 {
			mulScalarAVX2(dst[:limit], s)
		}
	} else if limit > 0 {
		mulScalarSSE2(dst[:limit], s)
	}

	for index := limit; index < len(dst); index++ {
		dst[index] *= s
	}
}

func reduceSumSq(a []float64) float64 {
	limit := alignedLen(len(a))
	sum := 0.0

	if useAVX2 {
		if limit > 0 {
			sum = reduceSumSqAVX2(a[:limit])
		}
	} else if limit > 0 {
		sum = reduceSumSqSSE2(a[:limit])
	}

	for _, value := range a[limit:] {
		sum += value * value
	}

	return sum
}

//go:noescape
func signVecAVX2(dst, src []float64)

//go:noescape
func signVecSSE2(dst, src []float64)

//go:noescape
func outerRowAVX2(dst, b []float64, scale float64)

//go:noescape
func outerRowSSE2(dst, b []float64, scale float64)

func signVec(dst, src []float64) {
	limit := alignedLen(len(src))

	if useAVX2 {
		if limit > 0 {
			signVecAVX2(dst[:limit], src[:limit])
		}
	} else if limit > 0 {
		signVecSSE2(dst[:limit], src[:limit])
	}

	for index := limit; index < len(src); index++ {
		switch {
		case src[index] > 0:
			dst[index] = 1
		case src[index] < 0:
			dst[index] = -1
		default:
			dst[index] = 0
		}
	}
}

func outerRow(dst, b []float64, scale float64) {
	limit := alignedLen(len(b))

	if useAVX2 {
		if limit > 0 {
			outerRowAVX2(dst[:limit], b[:limit], scale)
		}
	} else if limit > 0 {
		outerRowSSE2(dst[:limit], b[:limit], scale)
	}

	for i := limit; i < len(b); i++ {
		dst[i] = scale * b[i]
	}
}

//go:noescape
func addScaledVecAVX2(dst, src []float64, scale float64)

//go:noescape
func addScaledVecSSE2(dst, src []float64, scale float64)

//go:noescape
func sqrtVecAVX2(dst, src []float64)

//go:noescape
func sqrtVecSSE2(dst, src []float64)

//go:noescape
func addScalarVecAVX2(dst []float64, scalar float64)

//go:noescape
func addScalarVecSSE2(dst []float64, scalar float64)

//go:noescape
func divVecAVX2(dst, a, b []float64)

//go:noescape
func divVecSSE2(dst, a, b []float64)

func l2NormSqAVX2(a []float64) float64 { return reduceSumSqAVX2(a) }
func l2NormSqSSE2(a []float64) float64 { return reduceSumSqSSE2(a) }

//go:noescape
func clampVecAVX2(dst []float64, lo, hi float64)

//go:noescape
func clampVecSSE2(dst []float64, lo, hi float64)

//go:noescape
func expVecAVX2(dst, src []float64)

//go:noescape
func expVecSSE2(dst, src []float64)

//go:noescape
func logVecAVX2(dst, src []float64)

//go:noescape
func logVecSSE2(dst, src []float64)

//go:noescape
func layerNormRowAVX2(out, row, weight, bias []float64, eps float64)

//go:noescape
func layerNormRowSSE2(out, row, weight, bias []float64, eps float64)

//go:noescape
func rmsNormRowAVX2(out, row, weight []float64, eps float64)

//go:noescape
func rmsNormRowSSE2(out, row, weight []float64, eps float64)

//go:noescape
func softmaxRowAVX2(row []float64)

//go:noescape
func softmaxRowSSE2(row []float64)

func softmaxRowSIMD(row []float64) {
	if useAVX2 && useFMA {
		softmaxRowAVX2(row)
		return
	}

	softmaxRowSSE2(row)
}

//go:noescape
func logSumExpRowAVX2(row []float64) float64

//go:noescape
func logSumExpRowSSE2(row []float64) float64

func logSumExpRowSIMD(row []float64) float64 {
	if useAVX2 && useFMA {
		return logSumExpRowAVX2(row)
	}

	return logSumExpRowSSE2(row)
}

func rmsNormRow(out, row, weight []float64, eps float64) {
	if useAVX2 && useFMA {
		rmsNormRowAVX2(out, row, weight, eps)
		return
	}

	rmsNormRowSSE2(out, row, weight, eps)
}

func layerNormRow(out, row, weight, bias []float64, eps float64) {
	if useAVX2 && useFMA {
		layerNormRowAVX2(out, row, weight, bias, eps)
		return
	}

	layerNormRowSSE2(out, row, weight, bias, eps)
}

// addScaledVec: dst[i] += scale * src[i]  (AXPY)
func addScaledVec(dst, src []float64, scale float64) {
	limit := alignedLen(len(src))

	if useAVX2 {
		if limit > 0 {
			addScaledVecAVX2(dst[:limit], src[:limit], scale)
		}
	} else if limit > 0 {
		addScaledVecSSE2(dst[:limit], src[:limit], scale)
	}

	for i := limit; i < len(src); i++ {
		dst[i] += scale * src[i]
	}
}

// sqrtVec: dst[i] = sqrt(src[i])
func sqrtVec(dst, src []float64) {
	limit := alignedLen(len(src))

	if useAVX2 {
		if limit > 0 {
			sqrtVecAVX2(dst[:limit], src[:limit])
		}
	} else if limit > 0 {
		sqrtVecSSE2(dst[:limit], src[:limit])
	}

	if limit < len(src) {
		scalarSqrtTailKernel(dst, src, limit)
	}
}

// addScalarVec: dst[i] += scalar
func addScalarVec(dst []float64, scalar float64) {
	limit := alignedLen(len(dst))

	if useAVX2 {
		if limit > 0 {
			addScalarVecAVX2(dst[:limit], scalar)
		}
	} else if limit > 0 {
		addScalarVecSSE2(dst[:limit], scalar)
	}

	for i := limit; i < len(dst); i++ {
		dst[i] += scalar
	}
}

// divVec: dst[i] = a[i] / b[i]
func divVec(dst, a, b []float64) {
	limit := alignedLen(len(a))

	if useAVX2 {
		if limit > 0 {
			divVecAVX2(dst[:limit], a[:limit], b[:limit])
		}
	} else if limit > 0 {
		divVecSSE2(dst[:limit], a[:limit], b[:limit])
	}

	for i := limit; i < len(a); i++ {
		dst[i] = a[i] / b[i]
	}
}

// l2NormSq: sum(a[i]^2)
func l2NormSq(a []float64) float64 {
	if useAVX2 {
		return l2NormSqAVX2(a)
	}
	return l2NormSqSSE2(a)
}

// expVec: dst[i] = exp(src[i]).  Fully SIMD via polynomial range-reduction.
func expVec(dst, src []float64) {
	limit := alignedLen(len(src))

	if useAVX2 {
		if limit > 0 {
			expVecAVX2(dst[:limit], src[:limit])
		}
	} else if limit > 0 {
		expVecSSE2(dst[:limit], src[:limit])
	}

	if limit < len(src) {
		scalarExpTailKernel(dst, src, limit)
	}
}

// logVec: dst[i] = log(src[i]).
func logVec(dst, src []float64) {
	limit := alignedLen(len(src))

	if useAVX2 {
		if limit > 0 {
			logVecAVX2(dst[:limit], src[:limit])
		}
	} else if limit > 0 {
		logVecSSE2(dst[:limit], src[:limit])
	}

	if limit < len(src) {
		scalarLogTailKernel(dst, src, limit)
	}
}

// clampVec: dst[i] = clamp(dst[i], lo, hi)
func clampVec(dst []float64, lo, hi float64) {
	limit := alignedLen(len(dst))

	if useAVX2 {
		if limit > 0 {
			clampVecAVX2(dst[:limit], lo, hi)
		}
	} else if limit > 0 {
		clampVecSSE2(dst[:limit], lo, hi)
	}

	for i := limit; i < len(dst); i++ {
		if dst[i] < lo {
			dst[i] = lo
		} else if dst[i] > hi {
			dst[i] = hi
		}
	}
}

func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}
