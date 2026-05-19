package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalNormalizationDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalNormalizationDTypes {
		registerMetalLayerNormKernel(storageDType)
		registerMetalRMSNormKernel(storageDType)
		registerMetalGroupNormKernel(storageDType)
		registerMetalInstanceNormKernel(storageDType)
		registerMetalBatchNormEvalKernel(storageDType)
	}
}

func registerMetalLayerNormKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "layernorm",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalLayerNormKernel,
	})
}

func registerMetalRMSNormKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "rmsnorm",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalRMSNormKernel,
	})
}

func registerMetalGroupNormKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "groupnorm",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalGroupNormKernel,
	})
}

func registerMetalInstanceNormKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "instancenorm",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalInstanceNormKernel,
	})
}

func registerMetalBatchNormEvalKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "batchnorm_eval",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalBatchNormEvalKernel,
	})
}

func runMetalLayerNormKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalLayerNorm(args[0], args[1], args[2], args[3])
}

func runMetalRMSNormKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalRMSNorm(args[0], args[1], args[2])
}

func runMetalGroupNormKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalGroupNorm(args[0], args[1], args[2], args[3])
}

func runMetalInstanceNormKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalInstanceNorm(args[0], args[1], args[2], args[3])
}

func runMetalBatchNormEvalKernel(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	return runMetalBatchNormEval(args[0], args[1], args[2], args[3], args[4], args[5])
}
