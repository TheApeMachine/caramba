//go:build arm64

package cpu

// MatmulFloat32Native dispatches to the NEON outer-product driver
// when cols is a multiple of 4 (the common case), falling back to a
// scalar tail loop for the remaining columns.
func MatmulFloat32Native(out, left, right []float32, rows, inner, cols int) {
	colsBlock := cols & ^3
	tailStart := colsBlock

	if colsBlock > 0 {
		for row := 0; row < rows; row++ {
			MatmulRowFloat32NEONAsm(
				&out[row*cols],
				&left[row*inner],
				&right[0],
				inner,
				colsBlock,
			)
		}
	}

	// Scalar tail for cols % 4 != 0.
	if tailStart == cols {
		return
	}

	for row := 0; row < rows; row++ {
		for k := 0; k < inner; k++ {
			leftValue := left[row*inner+k]
			for col := tailStart; col < cols; col++ {
				out[row*cols+col] += leftValue * right[k*cols+col]
			}
		}
	}
}
