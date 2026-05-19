//go:build arm64

package cpu

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
