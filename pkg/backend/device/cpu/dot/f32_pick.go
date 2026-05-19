package dot

type f32DotKernelImpl struct {
	kernel    func(left, right *float32, count int) float32
	name      string
	available bool
}

func pickF32DotKernel(
	candidates []f32DotKernelImpl,
) func(left, right *float32, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("dot: no float32 dot kernel available")
}

type bf16DotKernelImpl struct {
	kernel    func(left, right *uint16, count int) uint16
	name      string
	available bool
}

func pickBF16DotKernel(
	candidates []bf16DotKernelImpl,
) func(left, right *uint16, count int) uint16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("dot: no bfloat16 dot kernel available")
}

type fp16DotKernelImpl struct {
	kernel    func(left, right *uint16, count int) uint16
	name      string
	available bool
}

func pickFP16DotKernel(
	candidates []fp16DotKernelImpl,
) func(left, right *uint16, count int) uint16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("dot: no float16 dot kernel available")
}

type int8DotKernelImpl struct {
	kernel    func(left, right *int8, count int) int32
	name      string
	available bool
}

func pickInt8DotKernel(
	candidates []int8DotKernelImpl,
) func(left, right *int8, count int) int32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("dot: no int8 dot kernel available")
}
