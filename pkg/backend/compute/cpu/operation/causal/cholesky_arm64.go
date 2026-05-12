//go:build arm64

package causal

import "fmt"

//go:noescape
func choleskyRegNEON(L []float64, n int, eps float64)

// choleskyReg performs an in-place regularised Cholesky factorisation on the
// n×n matrix stored row-major in L (length n*n). If a diagonal pivot becomes
// non-positive, it is clamped to eps instead of panicking. The lower triangle
// of L is overwritten with the Cholesky factor; the upper triangle is left
// untouched.
//
// Delegates to choleskyRegNEON, the ARM64 assembly implementation.
//
// Preconditions (caller-enforced):
//   - n >= 0
//   - eps > 0
//   - len(L) >= n*n
func choleskyReg(L []float64, n int, eps float64) {
	if n < 0 {
		panic(fmt.Sprintf("causal: choleskyReg: n must be >= 0, got %d", n))
	}

	if n == 0 {
		return
	}

	if !(eps > 0) {
		panic(fmt.Sprintf("causal: choleskyReg: eps must be > 0, got %g", eps))
	}

	if len(L) < n*n {
		panic(fmt.Sprintf("causal: choleskyReg: len(L)=%d < n*n=%d", len(L), n*n))
	}

	choleskyRegNEON(L, n, eps)
}
