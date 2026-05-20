package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

var metalHawkesMarkovDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalHawkesMarkovDTypes {
		registerMetalHawkesMarkovKernels(storageDType)
	}
}

func registerMetalHawkesMarkovKernels(storageDType dtype.DType) {
	registerMetalHawkesIntensityKernel(storageDType)
	registerMetalHawkesKernelMatrixKernel(storageDType)
	registerMetalHawkesLogLikelihoodKernel(storageDType)
	registerMetalMarkovMutualInformationKernel(storageDType)
	registerMetalMarkovBlanketPartitionKernel(storageDType)
	registerMetalMarkovFlowKernel("markov_flow_active", storageDType, runMetalMarkovFlowActiveKernel)
	registerMetalMarkovFlowKernel("markov_flow_internal", storageDType, runMetalMarkovFlowInternalKernel)
}

func registerMetalHawkesIntensityKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "hawkes_intensity",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalHawkesIntensityKernel,
	})
}

func registerMetalHawkesKernelMatrixKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "hawkes_kernel_matrix",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalHawkesKernelMatrixKernel,
	})
}

func registerMetalHawkesLogLikelihoodKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "hawkes_log_likelihood",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalHawkesLogLikelihoodKernel,
	})
}

func registerMetalMarkovMutualInformationKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "markov_mutual_information",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalMarkovMutualInformationKernel,
	})
}

func registerMetalMarkovBlanketPartitionKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "markov_blanket_partition",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32},
			Outputs: []dtype.DType{dtype.Int32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalMarkovBlanketPartitionKernel,
	})
}

func registerMetalMarkovFlowKernel(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
	})
}

func runMetalHawkesIntensityKernel(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	return runMetalHawkesIntensity(args[0], args[1], args[2], args[3], args[4], args[5])
}

func runMetalHawkesKernelMatrixKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalHawkesKernelMatrix(args[0], args[1], args[2], args[3])
}

func runMetalHawkesLogLikelihoodKernel(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	return runMetalHawkesLogLikelihood(args[0], args[1], args[2], args[3], args[4], args[5])
}

func runMetalMarkovMutualInformationKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalMarkovMutualInformation(args[0], args[1])
}

func runMetalMarkovBlanketPartitionKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalMarkovBlanketPartition(args[0], args[1], args[2])
}

func runMetalMarkovFlowActiveKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalMarkovFlow(args[0], args[1], args[2], 2)
}

func runMetalMarkovFlowInternalKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalMarkovFlow(args[0], args[1], args[2], 0)
}
