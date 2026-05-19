//go:build amd64

package matmul

import "golang.org/x/sys/cpu"

func MatmulFloat32Native(out, left, right []float32, rows, inner, cols int) {
	if cpu.X86.HasAVX512F {
		MatmulFloat32AVX512(out, left, right, rows, inner, cols)

		return
	}

	matmulFloat32Scalar(out, left, right, rows, inner, cols)
}

func MatmulFloat64Native(out, left, right []float64, rows, inner, cols int) {
	matmulFloat64Scalar(out, left, right, rows, inner, cols)
}

func SparseCSRMatMulFloat32Native(
	outView, valuesView, rightView []float32,
	rowPtr, colIdx []int32,
	rows, cols int,
) {
	SparseCSRMatMulFloat32Scalar(
		outView, valuesView, rightView,
		rowPtr, colIdx,
		rows, cols,
	)
}

func matmulFloat32Scalar(out, left, right []float32, rows, inner, cols int) {
	for row := 0; row < rows; row++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := left[row*inner+innerIndex]

			for col := 0; col < cols; col++ {
				out[row*cols+col] += leftValue * right[innerIndex*cols+col]
			}
		}
	}
}

func matmulFloat64Scalar(out, left, right []float64, rows, inner, cols int) {
	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := left[rowIndex*inner+innerIndex]

			for colIndex := 0; colIndex < cols; colIndex++ {
				out[rowIndex*cols+colIndex] +=
					leftValue * right[innerIndex*cols+colIndex]
			}
		}
	}
}
