//go:build !amd64 && !arm64

package kernels

func dotFloat32Native(a, b []float32) float32 {
	var sum float64

	for index := range a {
		sum += float64(a[index]) * float64(b[index])
	}

	return float32(sum)
}
