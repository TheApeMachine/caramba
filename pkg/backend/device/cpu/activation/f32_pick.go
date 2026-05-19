package activation

import "github.com/theapemachine/caramba/pkg/backend/device/cpu/peel"

type f32KernelImpl struct {
	kernel    func(dst, src *float32, count int)
	name      string
	available bool
}

func pickF32Kernel(candidates []f32KernelImpl) func(dst, src *float32, count int) {
	var genericKernel func(dst, src *float32, count int)

	for _, candidate := range candidates {
		if candidate.name == "generic" {
			genericKernel = candidate.kernel
		}
	}

	for _, candidate := range candidates {
		if !candidate.available {
			continue
		}

		if candidate.name == "generic" {
			return candidate.kernel
		}

		return peel.WrapF32Unary(candidate.kernel, genericKernel, candidate.name)
	}

	panic("activation: no float32 kernel available")
}
