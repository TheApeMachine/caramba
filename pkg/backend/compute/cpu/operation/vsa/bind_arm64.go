//go:build arm64

package vsa

//go:noescape
func bindKernelNEON(dst, a, b []float64)

func bindKernel(dst, a, b []float64) {
	n := len(a)
	limit := n / 2 * 2

	if limit > 0 {
		bindKernelNEON(dst[:limit], a[:limit], b[:limit])
	}

	if n%2 != 0 {
		dst[n-1] = a[n-1] * b[n-1]
	}
}
