package dot

var dotF32Kernel = func() func(left, right *float32, count int) float32 {
	return pickF32DotKernel(dotF32Funcs)
}()
