//go:build arm64

package hawkes

//go:noescape
func HawkesExpSumNEONAsm(exponents *float32, count int) float32

//go:noescape
func HawkesScaledExpStoreNEONAsm(
	exponents *float32,
	alpha float32,
	out *float32,
	count int,
)

/*
HawkesExpSumNEON sums vectorized exp lanes with a scalar epilogue for 1–3 tails.
*/
func HawkesExpSumNEON(exponents []float32, count int) float32 {
	if count == 0 {
		return 0
	}

	blockCount := count &^ 3
	sum := float32(0)

	if blockCount > 0 {
		sum = HawkesExpSumNEONAsm(&exponents[0], blockCount)
	}

	for index := blockCount; index < count; index++ {
		sum += hawkesExpScalar(exponents[index])
	}

	return sum
}

/*
HawkesScaledExpStoreNEON writes alpha*exp(exponents) with a scalar epilogue for 1–3 tails.
*/
func HawkesScaledExpStoreNEON(
	exponents []float32,
	alpha float32,
	out []float32,
	count int,
) {
	if count == 0 {
		return
	}

	blockCount := count &^ 3

	if blockCount > 0 {
		HawkesScaledExpStoreNEONAsm(&exponents[0], alpha, &out[0], blockCount)
	}

	for index := blockCount; index < count; index++ {
		out[index] = alpha * hawkesExpScalar(exponents[index])
	}
}
