package shape

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func init() {
	registerSliceReducedPrecision()
}

func registerSliceReducedPrecision() {
	for _, paramDType := range []dtype.DType{dtype.BFloat16, dtype.Float16} {
		paramDType := paramDType

		kernels.Default.Register(kernels.Kernel{
			Name: "slice",
			Signature: kernels.Signature{
				Layout: tensor.LayoutDense,
				Inputs: []dtype.DType{
					paramDType,
					dtype.Int32,
					dtype.Int32,
					dtype.Int32,
				},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runSliceMixed(args, paramDType)
			},
		})
	}
}
