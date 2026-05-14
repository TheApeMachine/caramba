//go:build arm64

package vsa

//go:noescape
func inversePermuteCopyNEON(dst, src []float64)

func inversePermuteKernel(dst, src []float64, shift int) {
	n := len(src)

	if n == 0 {
		return
	}

	k := ((shift % n) + n) % n
	split := n - k
	inversePermuteCopyNEON(dst[:split], src[k:])
	inversePermuteCopyNEON(dst[split:], src[:k])
}
