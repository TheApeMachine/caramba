package activation

type gatedTensorsKernelImpl struct {
	kernel    func(dst, gate, up *float32, count int)
	name      string
	available bool
}

func pickGatedTensorsKernel(
	candidates []gatedTensorsKernelImpl,
) func(dst, gate, up *float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("activation: no gated tensors kernel available")
}
