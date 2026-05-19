package reduction

type f32ReduceKernelImpl struct {
	kernel    func(values *float32, count int) float32
	name      string
	available bool
}

func pickF32ReduceKernel(
	candidates []f32ReduceKernelImpl,
) func(values *float32, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no float32 reduce kernel available")
}

type bf16SumKernelImpl struct {
	kernel    func(values *uint16, count int) uint16
	name      string
	available bool
}

func pickBF16SumKernel(
	candidates []bf16SumKernelImpl,
) func(values *uint16, count int) uint16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no bfloat16 sum kernel available")
}

type fp16SumKernelImpl struct {
	kernel    func(values *uint16, count int) uint16
	name      string
	available bool
}

func pickFP16SumKernel(
	candidates []fp16SumKernelImpl,
) func(values *uint16, count int) uint16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no float16 sum kernel available")
}
