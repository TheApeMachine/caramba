//go:build arm64

package attention

import "math"

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

func dotProduct(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func scaledAdd(dst, src []float64, scale float64) {
	for i := range dst {
		dst[i] += src[i] * scale
	}
}

func reduceMax(a []float64) float64 {
	m := math.Inf(-1)
	for _, v := range a {
		if v > m {
			m = v
		}
	}
	return m
}

func reduceSum(a []float64) float64 {
	var s float64
	for _, v := range a {
		s += v
	}
	return s
}

func divScalar(dst []float64, s float64) {
	for i := range dst {
		dst[i] /= s
	}
}
