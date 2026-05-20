package checkpoint

type float32DataEncodeKernelImpl struct {
	kernel    func(dst []byte, src []float32)
	name      string
	available bool
}

type float32DataDecodeKernelImpl struct {
	kernel    func(dst []float32, src []byte)
	name      string
	available bool
}

func pickEncodeFloat32DataKernel(
	candidates []float32DataEncodeKernelImpl,
) func(dst []byte, src []float32) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("checkpoint: no encode float32 data kernel available")
}

func pickDecodeFloat32DataKernel(
	candidates []float32DataDecodeKernelImpl,
) func(dst []float32, src []byte) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("checkpoint: no decode float32 data kernel available")
}
