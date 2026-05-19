package losses

var mseSumF32Kernel = func() func(predictions, targets *float32, count int) float32 {
	return pickF32PairSumKernel(mseSumF32Funcs)
}()

var maeSumF32Kernel = func() func(predictions, targets *float32, count int) float32 {
	return pickF32PairSumKernel(maeSumF32Funcs)
}()
