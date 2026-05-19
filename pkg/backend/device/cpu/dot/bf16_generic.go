package dot

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func DotBF16Generic(left, right *uint16, count int) uint16 {
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)
	var sum float32

	for index := 0; index < count; index++ {
		leftValue := dtype.BF16(leftView[index])
		rightValue := dtype.BF16(rightView[index])
		sum += (&leftValue).Float32() * (&rightValue).Float32()
	}

	return uint16(dtype.NewBfloat16FromFloat32(sum))
}
