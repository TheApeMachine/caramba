package masking

type f32ApplyMaskKernelImpl struct {
	kernel    func(input, mask, output *float32, count int)
	name      string
	available bool
}

type f32CausalMaskKernelImpl struct {
	kernel    func(output *float32, seqQ, seqK int)
	name      string
	available bool
}

type f32ALiBiBiasKernelImpl struct {
	kernel    func(scores, slope, output *float32, seqQ, seqK int)
	name      string
	available bool
}

func pickF32ApplyMaskKernel(
	candidates []f32ApplyMaskKernelImpl,
) func(input, mask, output *float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("masking: no float32 apply-mask kernel available")
}

func pickF32CausalMaskKernel(
	candidates []f32CausalMaskKernelImpl,
) func(output *float32, seqQ, seqK int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("masking: no float32 causal-mask kernel available")
}

func pickF32ALiBiBiasKernel(
	candidates []f32ALiBiBiasKernelImpl,
) func(scores, slope, output *float32, seqQ, seqK int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("masking: no float32 alibi-bias kernel available")
}
