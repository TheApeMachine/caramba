package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

var metalGeGLUTanhDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalGeGLUTanhDTypes {
		registerMetalGeGLUTanhKernel(storageDType)
	}
}

func registerMetalGeGLUTanhKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "geglu_tanh",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType,
				storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalGeGLUTanhKernel,
	})
}

func runMetalGeGLUTanhKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalGeGLUTanh(args[0], args[1], args[2])
}
