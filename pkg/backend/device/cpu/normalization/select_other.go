//go:build !amd64

package normalization

func NormSquaredDiffSumNative(row []float32, mean float32) float32 {
	return NormSquaredDiffSumGeneric(row, mean)
}

func NormApplyConstScaleBiasNative(
	outRow, row []float32,
	mean, invStdDev, scale, bias float32,
) {
	NormApplyConstScaleBiasGeneric(outRow, row, mean, invStdDev, scale, bias)
}
