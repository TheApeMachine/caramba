package activation

var (
	swiGLUPackedKernel = func() func(dst, packed *float32, batch, halfCount int) {
		return pickGatedPackedKernel(swiGLUPackedFuncs)
	}()

	linGLUPackedKernel = func() func(dst, packed *float32, batch, halfCount int) {
		return pickGatedPackedKernel(linGLUPackedFuncs)
	}()

	reGLUPackedKernel = func() func(dst, packed *float32, batch, halfCount int) {
		return pickGatedPackedKernel(reGLUPackedFuncs)
	}()

	gluPackedKernel = func() func(dst, packed *float32, batch, halfCount int) {
		return pickGatedPackedKernel(gluPackedFuncs)
	}()

	siGLUPackedKernel = func() func(dst, packed *float32, batch, halfCount int) {
		return pickGatedPackedKernel(siGLUPackedFuncs)
	}()

	seGLUPackedKernel = func() func(dst, packed *float32, batch, halfCount int) {
		return pickGatedPackedKernel(seGLUPackedFuncs)
	}()

	geGLUPackedKernel = func() func(dst, packed *float32, batch, halfCount int) {
		return pickGatedPackedKernel(geGLUPackedFuncs)
	}()

	geGLUTanhPackedKernel = func() func(dst, packed *float32, batch, halfCount int) {
		return pickGatedPackedKernel(geGLUTanhPackedFuncs)
	}()
)
