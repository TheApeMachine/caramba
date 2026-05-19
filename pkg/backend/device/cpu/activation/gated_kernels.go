package activation

var (
	swiGLUTensorsKernel = func() func(dst, gate, up *float32, count int) {
		return pickGatedTensorsKernel(swiGLUTensorsFuncs)
	}()

	linGLUTensorsKernel = func() func(dst, gate, up *float32, count int) {
		return pickGatedTensorsKernel(linGLUTensorsFuncs)
	}()

	reGLUTensorsKernel = func() func(dst, gate, up *float32, count int) {
		return pickGatedTensorsKernel(reGLUTensorsFuncs)
	}()

	gluTensorsKernel = func() func(dst, gate, up *float32, count int) {
		return pickGatedTensorsKernel(gluTensorsFuncs)
	}()

	siGLUTensorsKernel = func() func(dst, gate, up *float32, count int) {
		return pickGatedTensorsKernel(siGLUTensorsFuncs)
	}()

	seGLUTensorsKernel = func() func(dst, gate, up *float32, count int) {
		return pickGatedTensorsKernel(seGLUTensorsFuncs)
	}()

	geGLUTensorsKernel = func() func(dst, gate, up *float32, count int) {
		return pickGatedTensorsKernel(geGLUTensorsFuncs)
	}()

	geGLUTanhTensorsKernel = func() func(dst, gate, up *float32, count int) {
		return pickGatedTensorsKernel(geGLUTanhTensorsFuncs)
	}()
)
