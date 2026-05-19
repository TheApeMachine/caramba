package dot

import "unsafe"

var dotInt8Kernel = func() func(left, right *int8, count int) int32 {
	return pickInt8DotKernel(dotInt8Funcs)
}()

func dispatchDotInt8(left, right unsafe.Pointer, count int) int32 {
	if count == 0 {
		return 0
	}

	return dotInt8Kernel(
		(*int8)(left),
		(*int8)(right),
		count,
	)
}
