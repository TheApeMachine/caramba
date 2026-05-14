//go:build arm64

package math

func outerKernel(out, left, right []float64, rows, cols int) {
	for row := range rows {
		outputRow := out[row*cols : (row+1)*cols]
		outerRowNEON(outputRow, right, left[row])

		if cols%2 != 0 {
			col := cols - 1
			outputRow[col] = left[row] * right[col]
		}
	}
}
