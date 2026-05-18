package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalCausalDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalCausalDTypes {
		registerMetalCausalKernels(storageDType)
	}
}

func registerMetalCausalKernels(storageDType dtype.DType) {
	registerMetalCausalBinary("backdoor_adjustment", storageDType, runMetalBackdoorAdjustmentKernel)
	registerMetalCausalTernary("frontdoor_adjustment", storageDType, runMetalFrontdoorAdjustmentKernel)
	registerMetalCausalInt32("do_intervene", storageDType, runMetalDoInterveneKernel)
	registerMetalCausalBinary("cate", storageDType, runMetalCATEKernel)
	registerMetalCausalQuaternary("counterfactual", storageDType, runMetalCounterfactualKernel)
	registerMetalCausalTernary("iv_estimate", storageDType, runMetalIVEstimateKernel)
	registerMetalCausalInt32("dag_markov_factorization", storageDType, runMetalDAGMarkovFactorizationKernel)
}

func registerMetalCausalBinary(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	registerMetalCausalKernel(name, []dtype.DType{storageDType, storageDType}, storageDType, run)
}

func registerMetalCausalTernary(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	registerMetalCausalKernel(
		name, []dtype.DType{storageDType, storageDType, storageDType}, storageDType, run,
	)
}

func registerMetalCausalQuaternary(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	registerMetalCausalKernel(
		name,
		[]dtype.DType{storageDType, storageDType, storageDType, storageDType},
		storageDType,
		run,
	)
}

func registerMetalCausalInt32(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	registerMetalCausalKernel(name, []dtype.DType{storageDType, dtype.Int32}, storageDType, run)
}

func registerMetalCausalKernel(
	name string,
	inputDTypes []dtype.DType,
	outputDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  inputDTypes,
			Outputs: []dtype.DType{outputDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
	})
}

func runMetalBackdoorAdjustmentKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalBackdoorAdjustment(args[0], args[1], args[2])
}

func runMetalFrontdoorAdjustmentKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalFrontdoorAdjustment(args[0], args[1], args[2], args[3])
}

func runMetalDoInterveneKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalDoIntervene(args[0], args[1], args[2])
}

func runMetalCATEKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalCATE(args[0], args[1], args[2])
}

func runMetalCounterfactualKernel(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return runMetalCounterfactual(args[0], args[1], args[2], args[3], args[4])
}

func runMetalIVEstimateKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalIVEstimate(args[0], args[1], args[2], args[3])
}

func runMetalDAGMarkovFactorizationKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalDAGMarkovFactorization(args[0], args[1], args[2])
}
