package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalOptimizerDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalOptimizerDTypes {
		registerMetalOptimizerKernels(storageDType)
	}
}

func registerMetalOptimizerKernels(storageDType dtype.DType) {
	registerMetalOptimizer4Kernel("adam_step", storageDType, runMetalAdamStepKernel)
	registerMetalOptimizer4Kernel("adamw_step", storageDType, runMetalAdamWStepKernel)
	registerMetalOptimizer4Kernel("adamax_step", storageDType, runMetalAdamaxStepKernel)
	registerMetalOptimizer3Kernel("adagrad_step", storageDType, runMetalAdagradStepKernel)
	registerMetalOptimizer3Kernel("rmsprop_step", storageDType, runMetalRMSpropStepKernel)
	registerMetalOptimizer3Kernel("lion_step", storageDType, runMetalLionStepKernel)
	registerMetalOptimizer3Kernel("sgd_step", storageDType, runMetalSGDStepKernel)
	registerMetalOptimizer3Kernel("lars_step", storageDType, runMetalLARSStepKernel)
	registerMetalOptimizer2Kernel("lbfgs_step", storageDType, runMetalLBFGSStepKernel)
	registerMetalHebbianKernel(storageDType)
}

func registerMetalOptimizer4Kernel(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
	})
}

func registerMetalOptimizer3Kernel(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, dtype.Float32,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
	})
}

func registerMetalOptimizer2Kernel(
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

func registerMetalHebbianKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "hebbian_step",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalHebbianStepKernel,
	})
}

func runMetalAdamStepKernel(args ...tensor.Tensor) error {
	return runMetalOptimizer4Kernel(metalOptimizerAdam, args...)
}

func runMetalAdamWStepKernel(args ...tensor.Tensor) error {
	return runMetalOptimizer4Kernel(metalOptimizerAdamW, args...)
}

func runMetalAdamaxStepKernel(args ...tensor.Tensor) error {
	return runMetalOptimizer4Kernel(metalOptimizerAdamax, args...)
}

func runMetalAdagradStepKernel(args ...tensor.Tensor) error {
	return runMetalOptimizer3Kernel(metalOptimizerAdagrad, args...)
}

func runMetalRMSpropStepKernel(args ...tensor.Tensor) error {
	return runMetalOptimizer3Kernel(metalOptimizerRMSprop, args...)
}

func runMetalLionStepKernel(args ...tensor.Tensor) error {
	return runMetalOptimizer3Kernel(metalOptimizerLion, args...)
}

func runMetalSGDStepKernel(args ...tensor.Tensor) error {
	return runMetalOptimizer3Kernel(metalOptimizerSGD, args...)
}

func runMetalLARSStepKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalLARSStep(args[0], args[1], args[2], args[3])
}

func runMetalLBFGSStepKernel(args ...tensor.Tensor) error {
	return runMetalOptimizer2Kernel(metalOptimizerLBFGS, args...)
}

func runMetalHebbianStepKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalHebbianStep(args[0], args[1], args[2], args[3])
}
