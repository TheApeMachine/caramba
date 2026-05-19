//go:build !amd64 && !arm64

package cpu

func SumFloat32Native(values []float32) float32 {
	var sum float64

	for _, value := range values {
		sum += float64(value)
	}

	return float32(sum)
}
