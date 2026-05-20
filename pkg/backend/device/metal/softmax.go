package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

var metalSoftmaxDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalSoftmaxDTypes {
		registerMetalSoftmaxKernel(storageDType)
	}
}

func registerMetalSoftmaxKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "softmax",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalSoftmaxKernel,
	})
}

func runMetalSoftmaxKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalSoftmax(args[0], args[1])
}
