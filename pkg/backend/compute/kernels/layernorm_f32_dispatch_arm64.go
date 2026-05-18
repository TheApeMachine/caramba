//go:build arm64

package kernels

func layerNormApplyRowNative(out, row, scale, bias []float32, mean, invStdDev float32) {
	if len(out) == 0 {
		return
	}

	layerNormApplyRowNEONAsm(&out[0], &row[0], &scale[0], &bias[0], len(out), mean, invStdDev)
}

func layerNormSquaredDiffSumNative(row []float32, mean float32) float32 {
	if len(row) == 0 {
		return 0
	}

	return layerNormSquaredDiffSumNEONAsm(&row[0], len(row), mean)
}
