//go:build amd64

package vsa

//go:noescape
func inversePermuteCopyAVX2(dst, src []float64)

//go:noescape
func inversePermuteCopySSE2(dst, src []float64)

func inversePermuteKernel(dst, src []float64, shift int) {
	n := len(src)

	if n == 0 {
		return
	}

	k := ((shift % n) + n) % n
	split := n - k

	if useAVX2 {
		inversePermuteCopyAVX2(dst[:split], src[k:])
		inversePermuteCopyAVX2(dst[split:], src[:k])
		return
	}

	inversePermuteCopySSE2(dst[:split], src[k:])
	inversePermuteCopySSE2(dst[split:], src[:k])
}
