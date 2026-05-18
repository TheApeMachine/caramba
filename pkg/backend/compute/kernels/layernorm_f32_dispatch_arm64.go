//go:build arm64

package kernels

func layerNormApplyRowNative(out, row, scale, bias []float32, mean, invStdDev float32) {
	if len(out) == 0 {
		return
	}

	blockN := len(out) & ^3

	if blockN > 0 {
		layerNormApplyRowNEONAsm(&out[0], &row[0], &scale[0], &bias[0], blockN, mean, invStdDev)
	}

	for index := blockN; index < len(out); index++ {
		normalized := (row[index] - mean) * invStdDev
		out[index] = normalized*scale[index] + bias[index]
	}
}

func layerNormSquaredDiffSumNative(row []float32, mean float32) float32 {
	if len(row) == 0 {
		return 0
	}

	return layerNormSquaredDiffSumNEONAsm(&row[0], len(row), mean)
}
