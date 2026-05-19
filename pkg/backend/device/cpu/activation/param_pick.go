package activation

import "github.com/theapemachine/caramba/pkg/backend/device/cpu/peel"

type paramSlopeKernelImpl struct {
	kernel    func(dst, src *float32, count int, slope float32)
	name      string
	available bool
}

type paramRangeKernelImpl struct {
	kernel    func(dst, src *float32, count int, minVal, maxVal float32)
	name      string
	available bool
}

type paramRReluKernelImpl struct {
	kernel    func(dst, src *float32, count int, lower, upper float32)
	name      string
	available bool
}

type paramIndexedKernelImpl struct {
	kernel    func(dst, src, slopes *float32, count int)
	name      string
	available bool
}

func pickParamSlopeKernel(
	candidates []paramSlopeKernelImpl,
) func(dst, src *float32, count int, slope float32) {
	var genericKernel func(dst, src *float32, count int, slope float32)

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

		return peel.WrapParamSlope(candidate.kernel, genericKernel, candidate.name)
	}

	panic("activation: no param slope kernel available")
}

func pickParamRangeKernel(
	candidates []paramRangeKernelImpl,
) func(dst, src *float32, count int, minVal, maxVal float32) {
	var genericKernel func(dst, src *float32, count int, minVal, maxVal float32)

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

		return peel.WrapParamRange(candidate.kernel, genericKernel, candidate.name)
	}

	panic("activation: no param range kernel available")
}

func pickParamRReluKernel(
	candidates []paramRReluKernelImpl,
) func(dst, src *float32, count int, lower, upper float32) {
	var genericKernel func(dst, src *float32, count int, lower, upper float32)

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

		return peel.WrapParamRRelu(candidate.kernel, genericKernel, candidate.name)
	}

	panic("activation: no param rrelu kernel available")
}

func pickParamIndexedKernel(
	candidates []paramIndexedKernelImpl,
) func(dst, src, slopes *float32, count int) {
	var genericKernel func(dst, src, slopes *float32, count int)

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

		return peel.WrapParamIndexed(candidate.kernel, genericKernel, candidate.name)
	}

	panic("activation: no param indexed kernel available")
}
