//go:build arm64

package kernels

func softmaxRowFillExpsNative(row []float32, outRow []float32, maxValue float32) float32 {
	if len(row) == 0 {
		return 0
	}

	return softmaxRowExpSumNEONAsm(&outRow[0], &row[0], maxValue, len(row))
}
