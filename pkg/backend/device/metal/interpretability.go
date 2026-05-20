package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func init() {
	registerMetalActivationSteerFloat32Kernel()
}

func registerMetalActivationSteerFloat32Kernel() {
	kernels.Default.Register(kernels.Kernel{
		Name: "activation_steer_float32",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32,
				dtype.Float32,
				dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalActivationSteerFloat32Kernel,
	})
}

func runMetalActivationSteerFloat32Kernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalActivationSteerFloat32(args[0], args[1], args[2], args[3])
}
