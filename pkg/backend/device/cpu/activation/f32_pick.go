package activation

type f32KernelImpl struct {
	kernel    func(dst, src *float32, count int)
	name      string
	available bool
}

func pickF32Kernel(candidates []f32KernelImpl) func(dst, src *float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("activation: no float32 kernel available")
}
