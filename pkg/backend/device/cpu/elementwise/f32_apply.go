package elementwise

import "unsafe"

func runAddF32(dst, left, right unsafe.Pointer, count int) {
	addF32Kernel(
		(*float32)(dst), (*float32)(left), (*float32)(right), count,
	)
}

func runSubF32(dst, left, right unsafe.Pointer, count int) {
	subF32Kernel(
		(*float32)(dst), (*float32)(left), (*float32)(right), count,
	)
}

func runMulF32(dst, left, right unsafe.Pointer, count int) {
	mulF32Kernel(
		(*float32)(dst), (*float32)(left), (*float32)(right), count,
	)
}

func runDivF32(dst, left, right unsafe.Pointer, count int) {
	divF32Kernel(
		(*float32)(dst), (*float32)(left), (*float32)(right), count,
	)
}

func runMaxF32(dst, left, right unsafe.Pointer, count int) {
	maxF32Kernel(
		(*float32)(dst), (*float32)(left), (*float32)(right), count,
	)
}

func runMinF32(dst, left, right unsafe.Pointer, count int) {
	minF32Kernel(
		(*float32)(dst), (*float32)(left), (*float32)(right), count,
	)
}

func runAbsF32(dst, src unsafe.Pointer, count int) {
	absF32Kernel((*float32)(dst), (*float32)(src), count)
}

func runNegF32(dst, src unsafe.Pointer, count int) {
	negF32Kernel((*float32)(dst), (*float32)(src), count)
}

func runSqrtF32(dst, src unsafe.Pointer, count int) {
	sqrtF32Kernel((*float32)(dst), (*float32)(src), count)
}

func runReluF32(dst, src unsafe.Pointer, count int) {
	reluF32Kernel((*float32)(dst), (*float32)(src), count)
}

func runAxpyF32(y, x unsafe.Pointer, count int, alpha float32) {
	axpyF32Kernel((*float32)(y), (*float32)(x), alpha, count)
}

func runAddF64(dst, left, right unsafe.Pointer, count int) {
	addF64Kernel(
		(*float64)(dst), (*float64)(left), (*float64)(right), count,
	)
}

func runUnaryScalarF32(
	dst, src unsafe.Pointer,
	count int,
	apply func(float32) float32,
) {
	if count == 0 {
		return
	}

	dstView := unsafe.Slice((*float32)(dst), count)
	srcView := unsafe.Slice((*float32)(src), count)

	for index, value := range srcView {
		dstView[index] = apply(value)
	}
}
