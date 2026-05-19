//go:build arm64

package cpu

// MatmulFloat64Native dispatches to the NEON outer-product driver when
// cols is a multiple of 2, falling back to a scalar tail loop otherwise.
func MatmulFloat64Native(out, left, right []float64, rows, inner, cols int) {
	colsBlock := cols & ^1
	tailStart := colsBlock

	if colsBlock > 0 {
		for row := 0; row < rows; row++ {
			MatmulRowFloat64NEONAsm(
				&out[row*cols],
				&left[row*inner],
				&right[0],
				inner,
				colsBlock,
			)
		}
	}

	if tailStart == cols {
		return
	}

	for row := 0; row < rows; row++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := left[row*inner+innerIndex]

			for col := tailStart; col < cols; col++ {
				out[row*cols+col] += leftValue * right[innerIndex*cols+col]
			}
		}
	}
}
