//go:build !arm64

package layernorm

func LayerNormSquaredDiffSumNative(row []float32, mean float32) float32 {
	var sum float32

	for _, value := range row {
		diff := value - mean
		sum += diff * diff
	}

	return sum
}

func LayerNormApplyRowNative(
	outRow, row, scale, bias []float32,
	mean, invStdDev float32,
) {
	for index, value := range row {
		outRow[index] = (value-mean)*invStdDev*scale[index] + bias[index]
	}
}
