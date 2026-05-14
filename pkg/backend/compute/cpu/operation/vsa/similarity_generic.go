//go:build !amd64 && !arm64

package vsa

func similarityKernel(a, b []float64) float64 {
	sum := 0.0

	for index := range a {
		sum += a[index] * b[index]
	}

	return sum
}
