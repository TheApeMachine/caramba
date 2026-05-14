//go:build amd64

package vsa

//go:noescape
func similarityKernelAVX2(a, b []float64) float64

//go:noescape
func similarityKernelSSE2(a, b []float64) float64

func similarityKernel(a, b []float64) float64 {
	n := len(a)
	limit := alignedLen(n)
	sum := 0.0

	if useAVX2 {
		if limit > 0 {
			sum = similarityKernelAVX2(a[:limit], b[:limit])
		}
	} else if limit > 0 {
		sum = similarityKernelSSE2(a[:limit], b[:limit])
	}

	for index := limit; index < n; index++ {
		sum += a[index] * b[index]
	}

	return sum
}
