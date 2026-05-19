package matmul

func SparseCSRMatMulFloat32Scalar(
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

		for nzIndex := rowStart; nzIndex < rowEnd; nzIndex++ {
			colInLeft := int(colIdx[nzIndex])
			value := valuesView[nzIndex]

			for colIndex := 0; colIndex < cols; colIndex++ {
				outView[rowIndex*cols+colIndex] +=
					value * rightView[colInLeft*cols+colIndex]
			}
		}
	}
}
