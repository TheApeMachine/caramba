//go:build !arm64

package cpu

func SparseCSRMatMulFloat32Native(
	outView, valuesView, rightView []float32,
	rowPtr, colIdx []int32,
	rows, cols int,
) {
	sparseCSRMatMulFloat32Scalar(
		outView, valuesView, rightView,
		rowPtr, colIdx,
		rows, cols,
	)
}
