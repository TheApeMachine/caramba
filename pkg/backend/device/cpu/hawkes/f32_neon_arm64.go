//go:build arm64

package hawkes

//go:noescape
func HawkesExpSumNEONAsm(exponents *float32, n int) float32

//go:noescape
func HawkesScaledExpStoreNEONAsm(
	exponents *float32,
	alpha float32,
	out *float32,
	n int,
)
