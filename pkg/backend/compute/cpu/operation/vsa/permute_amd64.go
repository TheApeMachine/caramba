//go:build amd64

package vsa

//go:noescape
func permuteCopyAVX2(dst, src []float64)

//go:noescape
func permuteCopySSE2(dst, src []float64)

func permuteKernel(dst, src []float64, shift int) {
	n := len(src)

	if n == 0 {
		return
	}

	k := ((shift % n) + n) % n

	if useAVX2 {
		permuteCopyAVX2(dst[k:], src[:n-k])
		permuteCopyAVX2(dst[:k], src[n-k:])
		return
	}

	permuteCopySSE2(dst[k:], src[:n-k])
	permuteCopySSE2(dst[:k], src[n-k:])
}
