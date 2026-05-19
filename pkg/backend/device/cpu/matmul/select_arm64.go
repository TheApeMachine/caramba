//go:build arm64

package matmul

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

func SparseCSRMatMulFloat32Native(
	outView, valuesView, rightView []float32,
	rowPtr, colIdx []int32,
	rows, cols int,
) {
	for index := range outView {
		outView[index] = 0
	}

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowStart := int(rowPtr[rowIndex])
		rowEnd := int(rowPtr[rowIndex+1])

		if rowStart == rowEnd {
			continue
		}

		outputRow := outView[rowIndex*cols : (rowIndex+1)*cols]

		for nzIndex := rowStart; nzIndex < rowEnd; nzIndex++ {
			colInLeft := int(colIdx[nzIndex])
			denseRow := rightView[colInLeft*cols : (colInLeft+1)*cols]

			SparseCSRMatMulRowSingleNzNEONAsm(
				&outputRow[0],
				valuesView[nzIndex],
				&denseRow[0],
				cols,
			)
		}
	}
}
