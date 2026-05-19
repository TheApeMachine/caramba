package activation

type gatedPackedKernelImpl struct {
	kernel    func(dst, packed *float32, batch, halfCount int)
	name      string
	available bool
}

func pickGatedPackedKernel(
	candidates []gatedPackedKernelImpl,
) func(dst, packed *float32, batch, halfCount int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("activation: no gated packed kernel available")
}
