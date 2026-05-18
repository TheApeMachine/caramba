package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalResearchDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

type metalResearchOp int

const (
	metalResearchVSABind metalResearchOp = iota
	metalResearchVSABundle
	metalResearchVSAPermute
	metalResearchVSAInversePermute
	metalResearchPCPredictionError
)

func init() {
	for _, storageDType := range metalResearchDTypes {
		registerMetalVSAKernels(storageDType)
		registerMetalPredictiveCodingKernels(storageDType)
	}
}

func registerMetalVSAKernels(storageDType dtype.DType) {
	registerMetalResearchBinary("vsa_bind", storageDType, runMetalVSABindKernel)
	registerMetalResearchBinary("vsa_bundle", storageDType, runMetalVSABundleKernel)
	registerMetalResearchUnary("vsa_permute", storageDType, runMetalVSAPermuteKernel)
	registerMetalResearchUnary("vsa_inverse_permute", storageDType, runMetalVSAInversePermuteKernel)
}

func registerMetalPredictiveCodingKernels(storageDType dtype.DType) {
	registerMetalResearchBinary("pc_prediction", storageDType, runMetalPCPredictionKernel)
	registerMetalResearchBinary("pc_prediction_error", storageDType, runMetalPCPredictionErrorKernel)
	registerMetalResearchTernary(
		"pc_update_representation",
		storageDType,
		runMetalPCUpdateRepresentationKernel,
	)
	registerMetalResearchTernary("pc_update_weights", storageDType, runMetalPCUpdateWeightsKernel)
}

func registerMetalResearchBinary(
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

func registerMetalResearchUnary(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
	})
}

func registerMetalResearchTernary(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
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
		Run:       run,
	})
}

func runMetalVSABindKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalResearchBinaryKernel(metalResearchVSABind, args[0], args[1], args[2])
}

func runMetalVSABundleKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalResearchBinaryKernel(metalResearchVSABundle, args[0], args[1], args[2])
}

func runMetalVSAPermuteKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalResearchUnaryKernel(metalResearchVSAPermute, args[0], args[1])
}

func runMetalVSAInversePermuteKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalResearchUnaryKernel(metalResearchVSAInversePermute, args[0], args[1])
}

func runMetalPCPredictionKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalPCPrediction(args[0], args[1], args[2])
}

func runMetalPCPredictionErrorKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalResearchBinaryKernel(
		metalResearchPCPredictionError,
		args[0],
		args[1],
		args[2],
	)
}

func runMetalPCUpdateRepresentationKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalPCUpdateRepresentation(args[0], args[1], args[2], args[3])
}

func runMetalPCUpdateWeightsKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalPCUpdateWeights(args[0], args[1], args[2], args[3])
}
