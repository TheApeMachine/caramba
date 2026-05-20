package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

var metalProjectionDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalProjectionDTypes {
		registerMetalProjectionKernels(storageDType)
	}
}

func registerMetalProjectionKernels(storageDType dtype.DType) {
	registerMetalLinearKernel(storageDType)
	registerMetalFusedQKVKernel(storageDType)
	registerMetalLoRAMergeKernel(storageDType)
	registerMetalLoRAApplyKernel(storageDType)
}

func registerMetalLinearKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "linear",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalLinearKernel,
	})
}

func registerMetalFusedQKVKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "fused_qkv",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalFusedQKVKernel,
	})
}

func registerMetalLoRAMergeKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "lora_merge",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalLoRAMergeKernel,
	})
}

func registerMetalLoRAApplyKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "lora_apply",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalLoRAApplyKernel,
	})
}

func runMetalLinearKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalLinear(args[0], args[1], args[2], args[3])
}

func runMetalFusedQKVKernel(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	return runMetalFusedQKV(args[0], args[1], args[2], args[3], args[4], args[5])
}

func runMetalLoRAMergeKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalLoRAMerge(args[0], args[1], args[2], args[3])
}

func runMetalLoRAApplyKernel(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return runMetalLoRAApply(args[0], args[1], args[2], args[3], args[4])
}
