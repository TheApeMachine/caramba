//go:build amd64

package math

import "golang.org/x/sys/cpu"

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
	if useAVX2 {
		return reduceSumAVX2(a)
	}
	return reduceSumSSE2(a)
}

func reduceMax(a []float64) float64 {
	if useAVX2 {
		return reduceMaxAVX2(a)
	}
	return reduceMaxSSE2(a)
}

func divScalar(dst []float64, s float64) {
	if useAVX2 {
		divScalarAVX2(dst, s)
	} else {
		divScalarSSE2(dst, s)
	}
}

func addVec(dst, a, b []float64) {
	if useAVX2 {
		addVecAVX2(dst, a, b)
	} else {
		addVecSSE2(dst, a, b)
	}
}

func mulVec(dst, a, b []float64) {
	if useAVX2 {
		mulVecAVX2(dst, a, b)
	} else {
		mulVecSSE2(dst, a, b)
	}
}

func mulScalar(dst []float64, s float64) {
	if useAVX2 {
		mulScalarAVX2(dst, s)
	} else {
		mulScalarSSE2(dst, s)
	}
}

func reduceSumSq(a []float64) float64 {
	if useAVX2 {
		return reduceSumSqAVX2(a)
	}
	return reduceSumSqSSE2(a)
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
	n := len(src)
	stride := 4
	if !useAVX2 {
		stride = 2
		signVecSSE2(dst, src)
	} else {
		signVecAVX2(dst, src)
	}
	for i := (n / stride) * stride; i < n; i++ {
		switch {
		case src[i] > 0:
			dst[i] = 1
		case src[i] < 0:
			dst[i] = -1
		}
	}
}

func outerRow(dst, b []float64, scale float64) {
	n := len(b)
	stride := 4
	if !useAVX2 {
		stride = 2
		outerRowSSE2(dst, b, scale)
	} else {
		outerRowAVX2(dst, b, scale)
	}
	for i := (n / stride) * stride; i < n; i++ {
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

// addScaledVec: dst[i] += scale * src[i]  (AXPY)
func addScaledVec(dst, src []float64, scale float64) {
	n, stride := len(src), 4
	if !useAVX2 {
		stride = 2
		addScaledVecSSE2(dst, src, scale)
	} else {
		addScaledVecAVX2(dst, src, scale)
	}
	for i := (n / stride) * stride; i < n; i++ {
		dst[i] += scale * src[i]
	}
}

// sqrtVec: dst[i] = sqrt(src[i])
func sqrtVec(dst, src []float64) {
	n, stride := len(src), 4
	if !useAVX2 {
		stride = 2
		sqrtVecSSE2(dst, src)
	} else {
		sqrtVecAVX2(dst, src)
	}
	scalarSqrtTail(dst, src, (n/stride)*stride)
}

// addScalarVec: dst[i] += scalar
func addScalarVec(dst []float64, scalar float64) {
	n, stride := len(dst), 4
	if !useAVX2 {
		stride = 2
		addScalarVecSSE2(dst, scalar)
	} else {
		addScalarVecAVX2(dst, scalar)
	}
	for i := (n / stride) * stride; i < n; i++ {
		dst[i] += scalar
	}
}

// divVec: dst[i] = a[i] / b[i]
func divVec(dst, a, b []float64) {
	n, stride := len(a), 4
	if !useAVX2 {
		stride = 2
		divVecSSE2(dst, a, b)
	} else {
		divVecAVX2(dst, a, b)
	}
	for i := (n / stride) * stride; i < n; i++ {
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

// clampVec: dst[i] = clamp(dst[i], lo, hi)
func clampVec(dst []float64, lo, hi float64) {
	n, stride := len(dst), 4
	if !useAVX2 {
		stride = 2
		clampVecSSE2(dst, lo, hi)
	} else {
		clampVecAVX2(dst, lo, hi)
	}
	for i := (n / stride) * stride; i < n; i++ {
		if dst[i] < lo {
			dst[i] = lo
		} else if dst[i] > hi {
			dst[i] = hi
		}
	}
}
