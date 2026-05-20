package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

var metalGeGLUDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalGeGLUDTypes {
		registerMetalGeGLUKernel(storageDType)
	}
}

func registerMetalGeGLUKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "geglu",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType,
				storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalGeGLUKernel,
	})
}

func runMetalGeGLUKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalGeGLU(args[0], args[1], args[2])
}
