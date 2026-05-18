//go:build !amd64 && !arm64

package convert

func float32ToFloat64Native(dst []float64, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = float64(value)
	}

	return nil
}

func float64ToFloat32Native(dst []float32, src []float64) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = float32(value)
	}

	return nil
}
