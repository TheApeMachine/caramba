//go:build amd64

package math

func logKernel(out, input []float64) {
	width := 2
	vectorLog := logVecSSE2

	if useAVX2 {
		width = 4
		vectorLog = logVecAVX2
	}

	limit := len(input) / width * width

	if limit > 0 {
		vectorLog(out[:limit], input[:limit])
	}

	if limit < len(input) {
		scalarLogTailKernel(out, input, limit)
	}
}
