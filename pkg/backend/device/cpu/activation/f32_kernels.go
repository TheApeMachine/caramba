package activation

var (
	expF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(expF32Funcs)
	}()

	logF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(logF32Funcs)
	}()

	log1pF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(log1pF32Funcs)
	}()

	expm1F32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(expm1F32Funcs)
	}()

	sigmoidF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(sigmoidF32Funcs)
	}()

	logSigmoidF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(logSigmoidF32Funcs)
	}()

	tanhF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(tanhF32Funcs)
	}()

	siluF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(siluF32Funcs)
	}()

	geluTanhF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(geluTanhF32Funcs)
	}()

	geluF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(geluF32Funcs)
	}()

	reluF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(reluF32Funcs)
	}()

	leakyReluF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(leakyReluF32Funcs)
	}()

	eluF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(eluF32Funcs)
	}()

	celuF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(celuF32Funcs)
	}()

	seluF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(seluF32Funcs)
	}()

	softplusF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(softplusF32Funcs)
	}()

	mishF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(mishF32Funcs)
	}()

	softsignF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(softsignF32Funcs)
	}()

	hardSigmoidF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(hardSigmoidF32Funcs)
	}()

	hardSwishF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(hardSwishF32Funcs)
	}()

	hardTanhF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(hardTanhF32Funcs)
	}()

	hardGeluF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(hardGeluF32Funcs)
	}()

	quickGeluF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(quickGeluF32Funcs)
	}()

	tanhShrinkF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(tanhShrinkF32Funcs)
	}()

	softmaxF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(softmaxF32Funcs)
	}()

	logSoftmaxF32Kernel = func() func(dst, src *float32, count int) {
		return pickF32Kernel(logSoftmaxF32Funcs)
	}()
)
