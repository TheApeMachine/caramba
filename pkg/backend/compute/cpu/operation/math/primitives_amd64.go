//go:build amd64

package math

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
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
