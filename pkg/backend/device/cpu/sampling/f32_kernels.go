package sampling

var greedySampleF32Kernel = func() func(logits *float32, count int) int32 {
	return pickF32GreedyKernel(greedySampleF32Funcs)
}()

var samplingSoftmaxRowF32Kernel = func() func(logits, out *float32, temperature float32, count int) {
	return pickF32SoftmaxRowKernel(samplingSoftmaxRowF32Funcs)
}()
