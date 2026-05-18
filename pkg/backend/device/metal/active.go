package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalActiveDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

type metalActiveScalarOp int

const (
	metalActiveFreeEnergy metalActiveScalarOp = iota
	metalActiveExpectedFreeEnergy
)

func init() {
	for _, storageDType := range metalActiveDTypes {
		registerMetalActiveInferenceKernels(storageDType)
	}
}

func registerMetalActiveInferenceKernels(storageDType dtype.DType) {
	registerMetalFreeEnergyKernel(storageDType)
	registerMetalExpectedFreeEnergyKernel(storageDType)
	registerMetalActiveBinaryKernel("belief_update", storageDType, runMetalBeliefUpdateKernel)
	registerMetalActiveBinaryKernel("precision_weight", storageDType, runMetalPrecisionWeightKernel)
}

func registerMetalFreeEnergyKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "free_energy",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType,
				storageDType,
				storageDType,
				storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalFreeEnergyKernel,
	})
}

func registerMetalExpectedFreeEnergyKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "expected_free_energy",
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
		Run:       runMetalExpectedFreeEnergyKernel,
	})
}

func registerMetalActiveBinaryKernel(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
	})
}

func runMetalFreeEnergyKernel(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return runMetalFreeEnergy(args[0], args[1], args[2], args[3], args[4])
}

func runMetalExpectedFreeEnergyKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalExpectedFreeEnergy(args[0], args[1], args[2], args[3])
}

func runMetalBeliefUpdateKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalBeliefUpdate(args[0], args[1], args[2])
}

func runMetalPrecisionWeightKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalPrecisionWeight(args[0], args[1], args[2])
}
