package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func init() {
	registerMetalQuantizationKernels()
}

func registerMetalQuantizationKernels() {
	kernels.Default.Register(kernels.Kernel{
		Name: "int8_dequant",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Int8},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalInt8DequantKernel,
	})

	kernels.Default.Register(kernels.Kernel{
		Name: "int4_dequant",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Int4},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalInt4DequantKernel,
	})

	kernels.Default.Register(kernels.Kernel{
		Name: "int8_quant",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Int8},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalInt8QuantKernel,
	})
}

func runMetalInt8DequantKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalInt8Dequant(args[0], args[1])
}

func runMetalInt4DequantKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalInt4Dequant(args[0], args[1])
}

func runMetalInt8QuantKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalInt8Quant(args[0], args[1])
}
