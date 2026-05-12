//go:build arm64

package causal

//go:noescape
func choleskyRegNEON(L []float64, n int, eps float64)

func choleskyReg(L []float64, n int, eps float64) {
	choleskyRegNEON(L, n, eps)
}
