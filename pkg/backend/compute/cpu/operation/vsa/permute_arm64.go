//go:build arm64

package vsa

//go:noescape
func permuteCopyNEON(dst, src []float64)

func permuteKernel(dst, src []float64, shift int) {
	n := len(src)

	if n == 0 {
		return
	}

	k := ((shift % n) + n) % n
	permuteCopyNEON(dst[k:], src[:n-k])
	permuteCopyNEON(dst[:k], src[n-k:])
}
