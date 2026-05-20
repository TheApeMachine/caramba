package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type metalSamplingOp int

const (
	metalSamplingGreedy metalSamplingOp = iota
	metalSamplingTopK
	metalSamplingTopP
)

var metalSamplingDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalSamplingDTypes {
		registerMetalSamplingKernel("greedy_sample", storageDType, metalSamplingGreedy)
		registerMetalSamplingKernel("topk_sample", storageDType, metalSamplingTopK)
		registerMetalSamplingKernel("topp_sample", storageDType, metalSamplingTopP)
	}
}

func registerMetalSamplingKernel(
	name string,
	storageDType dtype.DType,
	operation metalSamplingOp,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{dtype.Int32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalSamplingKernel(operation),
	})
}

func runMetalSamplingKernel(operation metalSamplingOp) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 2 {
			return tensor.ErrShapeMismatch
		}

		return runMetalSampling(operation, args[0], args[1])
	}
}
