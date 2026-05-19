//go:build arm64

package layernorm

func LayerNormSquaredDiffSumNative(row []float32, mean float32) float32 {
	elementCount := len(row)

	if elementCount == 0 {
		return 0
	}

	blockCount := elementCount &^ 3
	var sum float32

	if blockCount > 0 {
		sum = LayerNormSquaredDiffSumNEONAsm(&row[0], blockCount, mean)
	}

	for index := blockCount; index < elementCount; index++ {
		diff := row[index] - mean
		sum += diff * diff
	}

	return sum
}

func LayerNormApplyRowNative(
	outRow, row, scale, bias []float32,
	mean, invStdDev float32,
) {
	elementCount := len(row)

	if elementCount == 0 {
		return
	}

	blockCount := elementCount &^ 3

	if blockCount > 0 {
		LayerNormApplyRowNEONAsm(
			&outRow[0], &row[0], &scale[0], &bias[0],
			blockCount, mean, invStdDev,
		)
	}

	for index := blockCount; index < elementCount; index++ {
		outRow[index] = (row[index]-mean)*invStdDev*scale[index] + bias[index]
	}
}
