package interpretability

var activationSteerFloat32Kernel = func() func(
	dst, base, direction []float32,
	coefficient float32,
	count int,
) {
	return pickActivationSteerKernel(activationSteerFloat32Funcs)
}()
