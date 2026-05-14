//go:build arm64

package vsa

//go:noescape
func similarityKernelNEON(a, b []float64) float64

func similarityKernel(a, b []float64) float64 {
	n := len(a)
	limit := n / 2 * 2
	sum := 0.0

	if limit > 0 {
		sum = similarityKernelNEON(a[:limit], b[:limit])
	}

	if n%2 != 0 {
		sum += a[n-1] * b[n-1]
	}

	return sum
}
