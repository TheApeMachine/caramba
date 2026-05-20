package model_editing

var weightGraftAddFloat32Kernel = func() func(weights, injection []float32, count int) {
	return pickWeightGraftAddKernel(weightGraftAddFloat32Funcs)
}()
