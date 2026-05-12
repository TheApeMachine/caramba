//go:build amd64

package vsa

import (
	"fmt"
	"math"

	"golang.org/x/sys/cpu"
)

var useAVX2 bool

func init() { useAVX2 = cpu.X86.HasAVX2 }

//go:noescape
func bindAVX2(dst, a, b []float64)

//go:noescape
func bindSSE2(dst, a, b []float64)

//go:noescape
func dotReduceAVX2(a, b []float64) float64

//go:noescape
func dotReduceSSE2(a, b []float64) float64

//go:noescape
func addInPlaceAVX2(dst, src []float64)

//go:noescape
func addInPlaceSSE2(dst, src []float64)

//go:noescape
func mulScalarVecAVX2(dst []float64, s float64)

//go:noescape
func mulScalarVecSSE2(dst []float64, s float64)

//go:noescape
func reduceSumSqAVX2(a []float64) float64

//go:noescape
func reduceSumSqSSE2(a []float64) float64

func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}

func applyBind(dst, a, b []float64) {
	if len(a) != len(b) || len(dst) != len(a) {
		panic(fmt.Sprintf(
			"vsa: applyBind: need len(dst)==len(a)==len(b), got %d %d %d",
			len(dst), len(a), len(b),
		))
	}

	n := len(a)
	limit := alignedLen(n)

	if useAVX2 {
		if limit > 0 {
			bindAVX2(dst[:limit], a[:limit], b[:limit])
		}
	} else if limit > 0 {
		bindSSE2(dst[:limit], a[:limit], b[:limit])
	}

	for i := limit; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func applyDot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf(
			"vsa: applyDot: len(a)=%d len(b)=%d must match",
			len(a), len(b),
		))
	}

	n := len(a)
	limit := alignedLen(n)
	sum := 0.0

	if useAVX2 {
		if limit > 0 {
			sum = dotReduceAVX2(a[:limit], b[:limit])
		}
	} else if limit > 0 {
		sum = dotReduceSSE2(a[:limit], b[:limit])
	}

	for i := limit; i < n; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

func applyAddInPlace(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf(
			"vsa: applyAddInPlace: need len(dst)>=%d, got len(dst)=%d",
			len(src), len(dst),
		))
	}

	n := len(src)
	limit := alignedLen(n)

	if useAVX2 {
		if limit > 0 {
			addInPlaceAVX2(dst[:limit], src[:limit])
		}
	} else if limit > 0 {
		addInPlaceSSE2(dst[:limit], src[:limit])
	}

	for i := limit; i < n; i++ {
		dst[i] += src[i]
	}
}

func applyMulScalar(dst []float64, s float64) {
	n := len(dst)
	limit := alignedLen(n)

	if useAVX2 {
		if limit > 0 {
			mulScalarVecAVX2(dst[:limit], s)
		}
	} else if limit > 0 {
		mulScalarVecSSE2(dst[:limit], s)
	}

	for i := limit; i < n; i++ {
		dst[i] *= s
	}
}

func applyReduceSumSq(a []float64) float64 {
	limit := alignedLen(len(a))
	sum := 0.0

	if useAVX2 {
		if limit > 0 {
			sum = reduceSumSqAVX2(a[:limit])
		}
	} else if limit > 0 {
		sum = reduceSumSqSSE2(a[:limit])
	}

	for _, v := range a[limit:] {
		sum += v * v
	}

	return sum
}

func applyL2Normalize(dst []float64) {
	norm := math.Sqrt(applyReduceSumSq(dst))

	if norm > l2NormEpsilon {
		applyMulScalar(dst, 1.0/norm)
	}
}
