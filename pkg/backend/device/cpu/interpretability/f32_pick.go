package interpretability

type activationSteerKernelImpl struct {
	kernel    func(dst, base, direction []float32, coefficient float32, count int)
	name      string
	available bool
}

func pickActivationSteerKernel(
	candidates []activationSteerKernelImpl,
) func(dst, base, direction []float32, coefficient float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("interpretability: no activation steer kernel available")
}
