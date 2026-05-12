//go:build arm64

package bench

//go:noescape
func argmaxNEON(xs []float64) int

func argmaxImpl(xs []float64) int {
	return argmaxNEON(xs)
}
