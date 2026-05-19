package dot

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

var dotBF16Kernel = func() func(left, right *uint16, count int) uint16 {
	return pickBF16DotKernel(dotBF16Funcs)
}()

func dispatchDotBF16(left, right unsafe.Pointer, count int) dtype.BF16 {
	if count == 0 {
		return 0
	}

	return dtype.BF16(dotBF16Kernel(
		(*uint16)(left),
		(*uint16)(right),
		count,
	))
}
