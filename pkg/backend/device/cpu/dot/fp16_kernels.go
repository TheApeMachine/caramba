package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var dotFP16Kernel = func() func(left, right *uint16, count int) uint16 {
	return pickFP16DotKernel(dotFP16Funcs)
}()

func dispatchDotFP16(left, right unsafe.Pointer, count int) dtype.F16 {
	if count == 0 {
		return 0
	}

	return dtype.F16(dotFP16Kernel(
		(*uint16)(left),
		(*uint16)(right),
		count,
	))
}
