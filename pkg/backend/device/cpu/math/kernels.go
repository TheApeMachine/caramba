package math

type f32InvSqrtDimScaleKernelImpl struct {
	kernel    func(out, input []float32, dim int32)
	name      string
	available bool
}

type f32LogSumExpKernelImpl struct {
	kernel    func(input []float32, cols int, out []float32)
	name      string
	available bool
}

type f32OuterKernelImpl struct {
	kernel    func(left, right, out []float32)
	name      string
	available bool
}

func pickInvSqrtDimScaleKernel(
	implementations []f32InvSqrtDimScaleKernelImpl,
) func(out, input []float32, dim int32) {
	for _, implementation := range implementations {
		if implementation.available {
			return implementation.kernel
		}
	}

	return InvSqrtDimScaleGeneric
}

func pickLogSumExpKernel(
	implementations []f32LogSumExpKernelImpl,
) func(input []float32, cols int, out []float32) {
	for _, implementation := range implementations {
		if implementation.available {
			return implementation.kernel
		}
	}

	return LogSumExpGeneric
}

func pickOuterKernel(
	implementations []f32OuterKernelImpl,
) func(left, right, out []float32) {
	for _, implementation := range implementations {
		if implementation.available {
			return implementation.kernel
		}
	}

	return OuterGeneric
}

func InvSqrtDimScaleF32(out, input []float32, dim int32) {
	pickInvSqrtDimScaleKernel(invSqrtDimScaleF32Funcs)(out, input, dim)
}

func LogSumExpF32(input []float32, cols int, out []float32) {
	pickLogSumExpKernel(logSumExpF32Funcs)(input, cols, out)
}

func OuterF32(left, right, out []float32) {
	pickOuterKernel(outerF32Funcs)(left, right, out)
}
