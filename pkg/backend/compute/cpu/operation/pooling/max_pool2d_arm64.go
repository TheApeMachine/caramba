//go:build arm64

package pooling

//go:noescape
func reduceMaxNEON(a []float64) float64

//go:noescape
func reduceSumNEON(a []float64) float64

func kernelMax(a []float64) float64 {
	return reduceMaxNEON(a)
}

func kernelSum(a []float64) float64 {
	return reduceSumNEON(a)
}
