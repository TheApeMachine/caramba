package activation

var (
	leakyReLUSlopeF32Kernel = func() func(dst, src *float32, count int, slope float32) {
		return pickParamSlopeKernel(leakyReLUSlopeF32Funcs)
	}()

	preluF32Kernel = func() func(dst, src *float32, count int, slope float32) {
		return pickParamSlopeKernel(preluF32Funcs)
	}()

	thresholdF32Kernel = func() func(dst, src *float32, count int, threshold float32) {
		return pickParamSlopeKernel(thresholdF32Funcs)
	}()

	hardTanhRangeF32Kernel = func() func(dst, src *float32, count int, minVal, maxVal float32) {
		return pickParamRangeKernel(hardTanhRangeF32Funcs)
	}()

	eluAlphaF32Kernel = func() func(dst, src *float32, count int, alpha float32) {
		return pickParamSlopeKernel(eluAlphaF32Funcs)
	}()

	celuAlphaF32Kernel = func() func(dst, src *float32, count int, alpha float32) {
		return pickParamSlopeKernel(celuAlphaF32Funcs)
	}()

	hardShrinkF32Kernel = func() func(dst, src *float32, count int, lambda float32) {
		return pickParamSlopeKernel(hardShrinkF32Funcs)
	}()

	softShrinkF32Kernel = func() func(dst, src *float32, count int, lambda float32) {
		return pickParamSlopeKernel(softShrinkF32Funcs)
	}()

	snakeF32Kernel = func() func(dst, src *float32, count int, alpha float32) {
		return pickParamSlopeKernel(snakeF32Funcs)
	}()

	snakeParametricF32Kernel = func() func(dst, src *float32, count int, alpha, beta float32) {
		return pickParamRangeKernel(snakeParametricF32Funcs)
	}()

	rreluF32Kernel = func() func(dst, src *float32, count int, lower, upper float32) {
		return pickParamRReluKernel(rreluF32Funcs)
	}()

	preluVF32Kernel = func() func(dst, src, slopes *float32, count int) {
		return pickParamIndexedKernel(preluVF32Funcs)
	}()
)
