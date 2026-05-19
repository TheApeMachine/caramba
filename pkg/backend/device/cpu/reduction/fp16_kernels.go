package reduction

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

var sumFP16Kernel = func() func(values *uint16, count int) uint16 {
	return pickFP16SumKernel(sumFP16Funcs)
}()

func dispatchSumFP16(values unsafe.Pointer, count int) dtype.F16 {
	if count == 0 {
		return 0
	}

	return dtype.F16(sumFP16Kernel(
		(*uint16)(values),
		count,
	))
}

func SumFP16Generic(values *uint16, count int) uint16 {
	view := unsafe.Slice(values, count)
	var sum float32

	for index := 0; index < count; index++ {
		value := dtype.F16(view[index])
		sum += value.Float32()
	}

	return uint16(dtype.Fromfloat32(sum))
}
