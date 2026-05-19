package dot

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func Dot(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchDot(left, right, count, format)
}
