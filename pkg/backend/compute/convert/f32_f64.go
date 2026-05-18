package convert

/*
Float32ToFloat64 widens float32 to float64.
*/
func Float32ToFloat64(dst []float64, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = float64(value)
	}

	return nil
}

/*
Float64ToFloat32 narrows float64 to float32. Default Go conversion
follows IEEE round-to-nearest-even.
*/
func Float64ToFloat32(dst []float32, src []float64) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = float32(value)
	}

	return nil
}
