//go:build amd64

package vsa

//go:noescape
func bindKernelAVX2(dst, a, b []float64)

//go:noescape
func bindKernelSSE2(dst, a, b []float64)

func bindKernel(dst, a, b []float64) {
	n := len(a)
	limit := alignedLen(n)

	if useAVX2 {
		if limit > 0 {
			bindKernelAVX2(dst[:limit], a[:limit], b[:limit])
		}
	} else if limit > 0 {
		bindKernelSSE2(dst[:limit], a[:limit], b[:limit])
	}

	for index := limit; index < n; index++ {
		dst[index] = a[index] * b[index]
	}
}
