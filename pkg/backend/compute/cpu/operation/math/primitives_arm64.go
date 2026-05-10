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
	return reduceSumNEON(a)
}

func reduceMax(a []float64) float64 {
	return reduceMaxNEON(a)
}

func divScalar(dst []float64, s float64) {
	divScalarNEON(dst, s)
}

func addVec(dst, a, b []float64) {
	addVecNEON(dst, a, b)
}

func mulVec(dst, a, b []float64) {
	mulVecNEON(dst, a, b)
}

func mulScalar(dst []float64, s float64) {
	mulScalarNEON(dst, s)
}

func reduceSumSq(a []float64) float64 {
	return reduceSumSqNEON(a)
}
