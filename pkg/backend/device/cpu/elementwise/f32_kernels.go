package elementwise

var (
	addF32Kernel = func() func(dst, left, right *float32, count int) {
		return pickF32BinaryKernel(addF32Funcs)
	}()

	subF32Kernel = func() func(dst, left, right *float32, count int) {
		return pickF32BinaryKernel(subF32Funcs)
	}()

	mulF32Kernel = func() func(dst, left, right *float32, count int) {
		return pickF32BinaryKernel(mulF32Funcs)
	}()

	divF32Kernel = func() func(dst, left, right *float32, count int) {
		return pickF32BinaryKernel(divF32Funcs)
	}()

	maxF32Kernel = func() func(dst, left, right *float32, count int) {
		return pickF32BinaryKernel(maxF32Funcs)
	}()

	minF32Kernel = func() func(dst, left, right *float32, count int) {
		return pickF32BinaryKernel(minF32Funcs)
	}()

	absF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32UnaryKernel(absF32Funcs)
	}()

	negF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32UnaryKernel(negF32Funcs)
	}()

	sqrtF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32UnaryKernel(sqrtF32Funcs)
	}()

	reluF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32UnaryKernel(reluF32Funcs)
	}()

	axpyF32Kernel = func() func(y, x *float32, alpha float32, count int) {
		return pickF32AxpyKernel(axpyF32Funcs)
	}()

	addF64Kernel = func() func(dst, left, right *float64, count int) {
		return pickF64BinaryKernel(addF64Funcs)
	}()
)
