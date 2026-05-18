//go:build arm64

package kernels

//go:noescape
func hawkesExpSumNEONAsm(exponents *float32, n int) float32

//go:noescape
func hawkesScaledExpStoreNEONAsm(
	exponents *float32,
	alpha float32,
	out *float32,
	n int,
)
