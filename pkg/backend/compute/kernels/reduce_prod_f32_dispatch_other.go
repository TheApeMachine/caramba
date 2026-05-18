//go:build !arm64

package kernels

func reduceProdFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	product := float64(1)

	for _, value := range values {
		product *= float64(value)
	}

	return float32(product)
}
