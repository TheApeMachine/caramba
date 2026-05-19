//go:build !arm64

package cpu

func ReduceMinFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	minimum := values[0]
	for _, value := range values[1:] {
		if value < minimum {
			minimum = value
		}
	}

	return minimum
}
