package dropout

type f32DropoutKernelImpl struct {
	kernel func(
		dst, src *float32,
		count int,
		seedState *[4]uint32,
		keepProb float32,
	)
	name      string
	available bool
}

func pickF32DropoutKernel(
	candidates []f32DropoutKernelImpl,
) func(
	dst, src *float32,
	count int,
	seedState *[4]uint32,
	keepProb float32,
) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("dropout: no float32 dropout kernel available")
}
