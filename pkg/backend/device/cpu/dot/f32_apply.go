package dot

import "unsafe"

func runDotF32(left, right unsafe.Pointer, count int) float32 {
	return dotF32Kernel(
		(*float32)(left),
		(*float32)(right),
		count,
	)
}
