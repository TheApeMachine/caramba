//go:build !amd64 && !arm64

package vsa

func permuteKernel(dst, src []float64, shift int) {
	n := len(src)

	if n == 0 {
		return
	}

	k := ((shift % n) + n) % n
	copy(dst[k:], src[:n-k])
	copy(dst[:k], src[n-k:])
}
