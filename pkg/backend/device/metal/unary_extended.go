package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type metalExtendedUnaryOperation struct {
	name      string
	operation metalUnaryFloat32Operation
}

var metalExtendedUnaryOperations = []metalExtendedUnaryOperation{
	{name: "rsqrt", operation: metalUnaryFloat32Rsqrt},
	{name: "exp", operation: metalUnaryFloat32Exp},
	{name: "log", operation: metalUnaryFloat32Log},
	{name: "sin", operation: metalUnaryFloat32Sin},
	{name: "cos", operation: metalUnaryFloat32Cos},
	{name: "tanh", operation: metalUnaryFloat32Tanh},
	{name: "sigmoid", operation: metalUnaryFloat32Sigmoid},
	{name: "silu", operation: metalUnaryFloat32Silu},
	{name: "swish", operation: metalUnaryFloat32Swish},
	{name: "softsign", operation: metalUnaryFloat32Softsign},
	{name: "elu", operation: metalUnaryFloat32ELU},
	{name: "selu", operation: metalUnaryFloat32SELU},
	{name: "leaky_relu", operation: metalUnaryFloat32LeakyReLU},
	{name: "hardsigmoid", operation: metalUnaryFloat32HardSigmoid},
	{name: "hardswish", operation: metalUnaryFloat32HardSwish},
}

var metalExtendedUnaryDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalExtendedUnaryDTypes {
		registerExtendedUnaryDTypeKernels(storageDType)
	}
}

func registerExtendedUnaryDTypeKernels(storageDType dtype.DType) {
	for _, unaryOperation := range metalExtendedUnaryOperations {
		registerUnaryKernel(
			unaryOperation.name,
			storageDType,
			runExtendedUnaryElementwise(unaryOperation.operation),
		)
	}
}

func runExtendedUnaryElementwise(
	operation metalUnaryFloat32Operation,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 2 {
			return tensor.ErrShapeMismatch
		}

		return runMetalUnaryElementwise(operation, args[0], args[1])
	}
}
