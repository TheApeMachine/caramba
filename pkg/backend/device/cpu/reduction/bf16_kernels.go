package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var sumBF16Kernel = func() func(values *uint16, count int) uint16 {
	return pickBF16SumKernel(sumBF16Funcs)
}()

func dispatchSumBF16(values unsafe.Pointer, count int) dtype.BF16 {
	if count == 0 {
		return 0
	}

	return dtype.BF16(sumBF16Kernel(
		(*uint16)(values),
		count,
	))
}

func SumBF16Generic(values *uint16, count int) uint16 {
	view := unsafe.Slice(values, count)
	var sum float32

	for index := 0; index < count; index++ {
		value := dtype.BF16(view[index])
		sum += (&value).Float32()
	}

	return uint16(dtype.NewBfloat16FromFloat32(sum))
}
