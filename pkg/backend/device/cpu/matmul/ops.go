package matmul

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func Matmul(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
) {
	dispatchMatmul(out, left, right, rows, inner, cols, format, runMatmulF32)
}
