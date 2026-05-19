package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalUtilityFloatDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	registerMetalCheckpointKernels()
	registerMetalTokenizerKernels()

	for _, storageDType := range metalUtilityFloatDTypes {
		registerMetalWeightFreezeMaskKernel(storageDType)
	}
}

func registerMetalCheckpointKernels() {
	kernels.Default.Register(kernels.Kernel{
		Name: "checkpoint_encode_float32",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Uint8},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalCheckpointEncodeFloat32Kernel,
	})

	kernels.Default.Register(kernels.Kernel{
		Name: "checkpoint_decode_float32",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Uint8},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalCheckpointDecodeFloat32Kernel,
	})
}

func registerMetalTokenizerKernels() {
	kernels.Default.Register(kernels.Kernel{
		Name: "tokenizer_pack_int32",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Int32},
			Outputs: []dtype.DType{dtype.Int32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalTokenizerPackInt32Kernel,
	})
}

func registerMetalWeightFreezeMaskKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "weight_freeze_mask",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Bool, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalWeightFreezeMaskKernel,
	})
}

func runMetalCheckpointEncodeFloat32Kernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalCheckpointEncodeFloat32(args[0], args[1])
}

func runMetalCheckpointDecodeFloat32Kernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalCheckpointDecodeFloat32(args[0], args[1])
}

func runMetalTokenizerPackInt32Kernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalTokenizerPackInt32(args[0], args[1])
}

func runMetalWeightFreezeMaskKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalWeightFreezeMask(args[0], args[1], args[2])
}
