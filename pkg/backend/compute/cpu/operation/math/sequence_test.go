package math

func mathSequence(length int, scale, offset float64) []float64 {
	values := make([]float64, length)

	for index := range values {
		pattern := float64((index % 31) - 15)
		values[index] = offset + scale*pattern
	}

	return values
}
