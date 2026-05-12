//go:build amd64

package markov_blanket

import "fmt"

// choleskyDecompAVX2 performs an in-place Cholesky factorisation A = L·Lᵀ on
// the n×n matrix stored row-major in L (length n*n). The lower triangle of L
// is overwritten with the Cholesky factor; the upper triangle is left
// untouched. Returns 1 on success, 0 if any diagonal pivot becomes
// non-positive (i.e. A is not numerically positive-definite).
//
//go:noescape
func choleskyDecompAVX2(L []float64, n int) uint64

// choleskyDecompSSE2 — SSE2 fallback with identical semantics.
//
//go:noescape
func choleskyDecompSSE2(L []float64, n int) uint64

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

	var ok uint64
	if useAVX2 && useFMA {
		ok = choleskyDecompAVX2(L, n)
	} else {
		ok = choleskyDecompSSE2(L, n)
	}

	if ok == 0 {
		return fmt.Errorf("markov_blanket: choleskyDecomp: non-positive pivot at some column (n=%d); matrix may not be positive-definite", n)
	}

	return nil
}
