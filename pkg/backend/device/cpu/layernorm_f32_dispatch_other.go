//go:build !arm64

package cpu

func LayerNormApplyRowNative(out, row, scale, bias []float32, mean, invStdDev float32) {
	for index, value := range row {
		normalized := (value - mean) * invStdDev
		out[index] = normalized*scale[index] + bias[index]
	}
}

func LayerNormSquaredDiffSumNative(row []float32, mean float32) float32 {
	var sum float32
	for _, value := range row {
		delta := value - mean
		sum += delta * delta
	}
	return sum
}
