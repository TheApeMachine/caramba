package tokenizer

type int32PackKernelImpl struct {
	kernel    func(dst, src *int32, count int)
	name      string
	available bool
}

func pickInt32PackKernel(
	candidates []int32PackKernelImpl,
) func(dst, src *int32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("tokenizer: no int32 pack kernel available")
}
