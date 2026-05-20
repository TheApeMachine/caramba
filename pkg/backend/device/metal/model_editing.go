package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func init() {
	registerMetalWeightGraftAddFloat32Kernel()
}

func registerMetalWeightGraftAddFloat32Kernel() {
	kernels.Default.Register(kernels.Kernel{
		Name: "weight_graft_add_float32",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32,
				dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalWeightGraftAddFloat32Kernel,
	})
}

func runMetalWeightGraftAddFloat32Kernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalWeightGraftAddFloat32(args[0], args[1])
}
