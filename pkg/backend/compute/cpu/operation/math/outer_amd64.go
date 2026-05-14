//go:build amd64

package math

func outerKernel(out, left, right []float64, rows, cols int) {
	width := 2
	vectorOuter := outerRowSSE2

	if useAVX2 {
		width = 4
		vectorOuter = outerRowAVX2
	}

	limit := cols / width * width

	for row := range rows {
		outputRow := out[row*cols : (row+1)*cols]

		if limit > 0 {
			vectorOuter(outputRow[:limit], right[:limit], left[row])
		}

		for col := limit; col < cols; col++ {
			outputRow[col] = left[row] * right[col]
		}
	}
}
