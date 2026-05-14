//go:build !amd64 && !arm64

package vsa

func bindKernel(dst, a, b []float64) {
	for index := range a {
		dst[index] = a[index] * b[index]
	}
}
