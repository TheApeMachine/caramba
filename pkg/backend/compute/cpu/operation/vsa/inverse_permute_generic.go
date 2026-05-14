//go:build !amd64 && !arm64

package vsa

func inversePermuteKernel(dst, src []float64, shift int) {
	n := len(src)

	if n == 0 {
		return
	}

	k := ((shift % n) + n) % n
	split := n - k
	copy(dst[:split], src[k:])
	copy(dst[split:], src[:k])
}
