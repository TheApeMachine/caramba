//go:build amd64

package bench

//go:noescape
func argmaxAVX2(xs []float64) int

func argmaxImpl(xs []float64) int {
	return argmaxAVX2(xs)
}
