package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalMatMulDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalMatMulDTypes {
		registerMetalMatMulKernel(storageDType)
		registerMetalMatMulAddKernel(storageDType)
	}
}

func registerMetalMatMulKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "matmul",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalMatMulKernel,
	})
}

func registerMetalMatMulAddKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "matmul_add",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType,
				storageDType,
				storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalMatMulAddKernel,
	})
}

func runMetalMatMulKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalMatMul(args[0], args[1], args[2])
}

func runMetalMatMulAddKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalMatMulAdd(args[0], args[1], args[2], args[3])
}
