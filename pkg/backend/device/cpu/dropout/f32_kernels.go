package dropout

var dropoutF32Kernel = func() func(
	dst, src *float32,
	count int,
	seedState *[4]uint32,
	keepProb float32,
) {
	return pickF32DropoutKernel(dropoutF32Funcs)
}()
