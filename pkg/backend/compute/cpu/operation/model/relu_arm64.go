//go:build arm64

package model

//go:noescape
func reluNEON(dst []float64)

func reluInPlace(x []float64) {
	reluNEON(x)
}
