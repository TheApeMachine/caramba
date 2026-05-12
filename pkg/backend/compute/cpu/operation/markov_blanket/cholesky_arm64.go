//go:build arm64

package markov_blanket

//go:noescape
func choleskyDecompNEON(L []float64, n int) uint64

func choleskyDecomp(L []float64, n int) bool {
	return choleskyDecompNEON(L, n) == 1
}
