package activation

import "github.com/theapemachine/caramba/pkg/backend/device/cpu/peel"

type gatedTensorsKernelImpl struct {
	kernel    func(dst, gate, up *float32, count int)
	name      string
	available bool
}

func pickGatedTensorsKernel(
	candidates []gatedTensorsKernelImpl,
) func(dst, gate, up *float32, count int) {
	var genericKernel func(dst, gate, up *float32, count int)

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

		return peel.WrapGatedTensors(candidate.kernel, genericKernel, candidate.name)
	}

	panic("activation: no gated tensors kernel available")
}
