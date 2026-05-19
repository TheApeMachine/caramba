package matmul

import "unsafe"

func runMatmulF32(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
) {
	matmulF32Kernel(
		unsafe.Slice((*float32)(out), rows*cols),
		unsafe.Slice((*float32)(left), rows*inner),
		unsafe.Slice((*float32)(right), inner*cols),
		rows, inner, cols,
	)
}

func runMatmulF64(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
) {
	matmulF64Kernel(
		unsafe.Slice((*float64)(out), rows*cols),
		unsafe.Slice((*float64)(left), rows*inner),
		unsafe.Slice((*float64)(right), inner*cols),
		rows, inner, cols,
	)
}
