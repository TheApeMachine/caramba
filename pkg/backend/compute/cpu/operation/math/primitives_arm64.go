//go:build arm64

package math

//go:noescape
func reduceSumNEON(a []float64) float64

//go:noescape
func reduceMaxNEON(a []float64) float64

//go:noescape
func divScalarNEON(dst []float64, s float64)

//go:noescape
func addVecNEON(dst, a, b []float64)

//go:noescape
func mulVecNEON(dst, a, b []float64)

//go:noescape
func mulScalarNEON(dst []float64, s float64)

//go:noescape
func reduceSumSqNEON(a []float64) float64

func reduceSum(a []float64) float64 {
	n := len(a)
	sum := reduceSumNEON(a)
	if n%2 != 0 {
		sum += a[n-1]
	}
	return sum
}

func reduceMax(a []float64) float64 {
	n := len(a)
	mx := reduceMaxNEON(a)
	if n%2 != 0 {
		if a[n-1] > mx {
			mx = a[n-1]
		}
	}
	return mx
}

func divScalar(dst []float64, s float64) {
	n := len(dst)
	divScalarNEON(dst, s)
	if n%2 != 0 {
		dst[n-1] /= s
	}
}

func addVec(dst, a, b []float64) {
	n := len(a)
	addVecNEON(dst, a, b)
	if n%2 != 0 {
		dst[n-1] = a[n-1] + b[n-1]
	}
}

func mulVec(dst, a, b []float64) {
	n := len(a)
	mulVecNEON(dst, a, b)
	if n%2 != 0 {
		dst[n-1] = a[n-1] * b[n-1]
	}
}

func mulScalar(dst []float64, s float64) {
	n := len(dst)
	mulScalarNEON(dst, s)
	if n%2 != 0 {
		dst[n-1] *= s
	}
}

func reduceSumSq(a []float64) float64 {
	n := len(a)
	sum := reduceSumSqNEON(a)
	if n%2 != 0 {
		sum += a[n-1] * a[n-1]
	}
	return sum
}

//go:noescape
func signVecNEON(dst, src []float64)

//go:noescape
func outerRowNEON(dst, b []float64, scale float64)

func signVec(dst, src []float64) {
	n := len(src)
	signVecNEON(dst, src)
	if n%2 != 0 {
		v := src[n-1]
		if v > 0 {
			dst[n-1] = 1
		} else if v < 0 {
			dst[n-1] = -1
		} else {
			dst[n-1] = 0
		}
	}
}

func outerRow(dst, b []float64, scale float64) {
	n := len(b)
	outerRowNEON(dst, b, scale)
	// scalar tail for odd-length vectors
	if n%2 != 0 {
		dst[n-1] = scale * b[n-1]
	}
}

//go:noescape
func addScaledVecNEON(dst, src []float64, scale float64)

//go:noescape
func sqrtVecNEON(dst, src []float64)

//go:noescape
func addScalarVecNEON(dst []float64, scalar float64)

//go:noescape
func divVecNEON(dst, a, b []float64)

//go:noescape
func clampVecNEON(dst []float64, lo, hi float64)

//go:noescape
func expVecNEON(dst, src []float64)

//go:noescape
func logVecNEON(dst, src []float64)

func expVec(dst, src []float64) {
	expVecNEON(dst, src)
}

func logVec(dst, src []float64) {
	logVecNEON(dst, src)
}

func addScaledVec(dst, src []float64, scale float64) {
	n := len(src)
	addScaledVecNEON(dst, src, scale)
	if n%2 != 0 {
		dst[n-1] += scale * src[n-1]
	}
}

func sqrtVec(dst, src []float64) {
	n := len(src)
	sqrtVecNEON(dst, src)
	if n%2 != 0 {
		scalarSqrtTail(dst, src, n-1)
	}
}

func addScalarVec(dst []float64, scalar float64) {
	n := len(dst)
	addScalarVecNEON(dst, scalar)
	if n%2 != 0 {
		dst[n-1] += scalar
	}
}

func divVec(dst, a, b []float64) {
	n := len(a)
	divVecNEON(dst, a, b)
	if n%2 != 0 {
		dst[n-1] = a[n-1] / b[n-1]
	}
}

func l2NormSq(a []float64) float64 {
	return reduceSumSq(a)
}

func clampVec(dst []float64, lo, hi float64) {
	n := len(dst)
	clampVecNEON(dst, lo, hi)
	if n%2 != 0 {
		v := dst[n-1]
		if v < lo {
			dst[n-1] = lo
		} else if v > hi {
			dst[n-1] = hi
		}
	}
}
