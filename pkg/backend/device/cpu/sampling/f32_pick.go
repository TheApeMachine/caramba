package sampling

type f32GreedyKernelImpl struct {
	kernel    func(logits *float32, count int) int32
	name      string
	available bool
}

type f32SoftmaxRowKernelImpl struct {
	kernel    func(logits, out *float32, temperature float32, count int)
	name      string
	available bool
}

func pickF32GreedyKernel(
	candidates []f32GreedyKernelImpl,
) func(logits *float32, count int) int32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("sampling: no float32 greedy-sample kernel available")
}

func pickF32SoftmaxRowKernel(
	candidates []f32SoftmaxRowKernelImpl,
) func(logits, out *float32, temperature float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("sampling: no float32 softmax-row kernel available")
}
