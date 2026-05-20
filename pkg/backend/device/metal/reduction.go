package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

var metalReductionDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalReductionDTypes {
		registerMetalReductionKernels(storageDType)
	}
}

func registerMetalReductionKernels(storageDType dtype.DType) {
	registerMetalReductionKernel("sum", storageDType, runMetalReductionSumKernel)
	registerMetalReductionKernel("mean", storageDType, runMetalReductionMeanKernel)
	registerMetalReductionKernel("prod", storageDType, runMetalReductionProdKernel)
	registerMetalReductionKernel("reduce_min", storageDType, runMetalReductionMinKernel)
	registerMetalReductionKernel("reduce_max", storageDType, runMetalReductionMaxKernel)
	registerMetalReductionKernel("argmin", storageDType, runMetalReductionArgminKernel)
	registerMetalReductionKernel("argmax", storageDType, runMetalReductionArgmaxKernel)
	registerMetalReductionKernel("l1_norm", storageDType, runMetalReductionL1NormKernel)
	registerMetalReductionKernel("l2_norm", storageDType, runMetalReductionL2NormKernel)
	registerMetalReductionKernel("variance", storageDType, runMetalReductionVarianceKernel)
	registerMetalReductionKernel("stddev", storageDType, runMetalReductionStddevKernel)
}

func registerMetalReductionKernel(
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

func runMetalReductionSumKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionSum, args...)
}

func runMetalReductionMeanKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionMean, args...)
}

func runMetalReductionProdKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionProd, args...)
}

func runMetalReductionMinKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionMin, args...)
}

func runMetalReductionMaxKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionMax, args...)
}

func runMetalReductionArgminKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionArgmin, args...)
}

func runMetalReductionArgmaxKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionArgmax, args...)
}

func runMetalReductionL1NormKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionL1Norm, args...)
}

func runMetalReductionL2NormKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionL2Norm, args...)
}

func runMetalReductionVarianceKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionVariance, args...)
}

func runMetalReductionStddevKernel(args ...tensor.Tensor) error {
	return runMetalReductionKernel(metalReductionStddev, args...)
}
