package shape

type f32CopyContiguousKernelImpl struct {
	kernel    func(dst, src *float32, count int)
	name      string
	available bool
}

type f32WhereKernelImpl struct {
	kernel    func(dst, positive, negative *float32, mask []byte, count int)
	name      string
	available bool
}

type f32MaskedFillKernelImpl struct {
	kernel    func(dst, input *float32, fill float32, mask []byte, count int)
	name      string
	available bool
}

func pickF32CopyContiguousKernel(
	candidates []f32CopyContiguousKernelImpl,
) func(dst, src *float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("shape: no float32 copy-contiguous kernel available")
}

func pickF32WhereKernel(
	candidates []f32WhereKernelImpl,
) func(dst, positive, negative *float32, mask []byte, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("shape: no float32 where kernel available")
}

func pickF32MaskedFillKernel(
	candidates []f32MaskedFillKernelImpl,
) func(dst, input *float32, fill float32, mask []byte, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("shape: no float32 masked-fill kernel available")
}
