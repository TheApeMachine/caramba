package normalization

func NormSquaredDiffSumGeneric(row []float32, mean float32) float32 {
	var sum float32

	for _, value := range row {
		diff := value - mean
		sum += diff * diff
	}

	return sum
}

func NormApplyConstScaleBiasGeneric(
	outRow, row []float32,
	mean, invStdDev, scale, bias float32,
) {
	for index, value := range row {
		normalized := (value - mean) * invStdDev
		outRow[index] = normalized*scale + bias
	}
}
