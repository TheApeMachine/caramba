//go:build !arm64

package kernels

func reduceMaxFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	maximum := values[0]
	for _, value := range values[1:] {
		if value > maximum {
			maximum = value
		}
	}

	return maximum
}
