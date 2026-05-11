//go:build arm64

package vsa

import "math"

//go:noescape
func bindNEON(dst, a, b []float64)

//go:noescape
func dotReduceNEON(a, b []float64) float64

//go:noescape
func addInPlaceNEON(dst, src []float64)

//go:noescape
func mulScalarVecNEON(dst []float64, s float64)

//go:noescape
func reduceSumSqNEON(a []float64) float64

func applyBind(dst, a, b []float64) {
	n := len(a)
	limit := n / 2 * 2

	if limit > 0 {
		bindNEON(dst[:limit], a[:limit], b[:limit])
	}

	if n%2 != 0 {
		dst[n-1] = a[n-1] * b[n-1]
	}
}

func applyDot(a, b []float64) float64 {
	n := len(a)
	limit := n / 2 * 2
	sum := 0.0

	if limit > 0 {
		sum = dotReduceNEON(a[:limit], b[:limit])
	}

	if n%2 != 0 {
		sum += a[n-1] * b[n-1]
	}

	return sum
}

func applyAddInPlace(dst, src []float64) {
	n := len(src)
	limit := n / 2 * 2

	if limit > 0 {
		addInPlaceNEON(dst[:limit], src[:limit])
	}

	if n%2 != 0 {
		dst[n-1] += src[n-1]
	}
}

func applyMulScalar(dst []float64, s float64) {
	n := len(dst)
	limit := n / 2 * 2

	if limit > 0 {
		mulScalarVecNEON(dst[:limit], s)
	}

	if n%2 != 0 {
		dst[n-1] *= s
	}
}

func applyReduceSumSq(a []float64) float64 {
	n := len(a)
	limit := n / 2 * 2
	sum := 0.0

	if limit > 0 {
		sum = reduceSumSqNEON(a[:limit])
	}

	if n%2 != 0 {
		sum += a[n-1] * a[n-1]
	}

	return sum
}

func applyL2Normalize(dst []float64) {
	norm := math.Sqrt(applyReduceSumSq(dst))

	if norm > 1e-12 {
		applyMulScalar(dst, 1.0/norm)
	}
}
