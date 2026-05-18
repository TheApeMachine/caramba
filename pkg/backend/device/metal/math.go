package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalMathDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalMathDTypes {
		registerMetalMathKernels(storageDType)
	}
}

func registerMetalMathKernels(storageDType dtype.DType) {
	registerMetalInvSqrtDimScaleKernel(storageDType)
	registerMetalLogSumExpKernel(storageDType)
	registerMetalOuterKernel(storageDType)
}

func registerMetalInvSqrtDimScaleKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "inv_sqrt_dim_scale",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalInvSqrtDimScaleKernel,
	})
}

func registerMetalLogSumExpKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "logsumexp",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalLogSumExpKernel,
	})
}

func registerMetalOuterKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "outer",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalOuterKernel,
	})
}

func runMetalInvSqrtDimScaleKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalInvSqrtDimScale(args[0], args[1], args[2])
}

func runMetalLogSumExpKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalLogSumExp(args[0], args[1])
}

func runMetalOuterKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalOuter(args[0], args[1], args[2])
}
