//go:build arm64

package markov_blanket

import "fmt"

// choleskyDecompNEON performs an in-place Cholesky factorisation A = L·Lᵀ on
// the n×n matrix stored row-major in L (length n*n). The lower triangle of L
// is overwritten with the Cholesky factor; the upper triangle is left
// untouched. Returns 1 on success, 0 if any diagonal pivot becomes
// non-positive.
//
//go:noescape
func choleskyDecompNEON(L []float64, n int) uint64

// choleskyDecomp performs an in-place Cholesky factorisation. Returns nil on
// success, or an error describing the failure (invalid input, non-PD).
//
// Preconditions (caller-enforced):
//   - n >= 0
//   - len(L) >= n*n
func choleskyDecomp(L []float64, n int) error {
	if n < 0 {
		return fmt.Errorf("markov_blanket: choleskyDecomp: n must be >= 0, got %d", n)
	}

	if n == 0 {
		return nil
	}

	if len(L) < n*n {
		return fmt.Errorf("markov_blanket: choleskyDecomp: len(L)=%d < n*n=%d", len(L), n*n)
	}

	if choleskyDecompNEON(L, n) == 0 {
		return fmt.Errorf("markov_blanket: choleskyDecomp: non-positive pivot at some column (n=%d); matrix may not be positive-definite", n)
	}

	return nil
}
