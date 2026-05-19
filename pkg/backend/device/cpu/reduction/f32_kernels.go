package reduction

var (
	sumF32Kernel = func() func(values *float32, count int) float32 {
		return pickF32ReduceKernel(sumF32Funcs)
	}()

	prodF32Kernel = func() func(values *float32, count int) float32 {
		return pickF32ReduceKernel(prodF32Funcs)
	}()

	minF32Kernel = func() func(values *float32, count int) float32 {
		return pickF32ReduceKernel(minF32Funcs)
	}()

	maxF32Kernel = func() func(values *float32, count int) float32 {
		return pickF32ReduceKernel(maxF32Funcs)
	}()

	l1NormF32Kernel = func() func(values *float32, count int) float32 {
		return pickF32ReduceKernel(l1NormF32Funcs)
	}()
)
