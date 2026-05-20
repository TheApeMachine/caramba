package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

var metalLossDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalLossDTypes {
		registerMetalLossKernels(storageDType)
	}
}

func registerMetalLossKernels(storageDType dtype.DType) {
	registerMetalPairLossKernel("mse_loss", storageDType, runMetalMSELossKernel)
	registerMetalPairLossKernel("mae_loss", storageDType, runMetalMAELossKernel)
	registerMetalPairLossKernel("huber_loss", storageDType, runMetalHuberLossKernel)
	registerMetalPairLossKernel(
		"binary_cross_entropy", storageDType, runMetalBinaryCrossEntropyLossKernel,
	)
	registerMetalPairLossKernel("kl_divergence", storageDType, runMetalKLDivergenceKernel)
	registerMetalCrossEntropyLossKernel(storageDType)
}

func registerMetalPairLossKernel(
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

func registerMetalCrossEntropyLossKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "cross_entropy",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalCrossEntropyLossKernel,
	})
}

func runMetalMSELossKernel(args ...tensor.Tensor) error {
	return runMetalPairLossKernel(metalLossMSE, args...)
}

func runMetalMAELossKernel(args ...tensor.Tensor) error {
	return runMetalPairLossKernel(metalLossMAE, args...)
}

func runMetalHuberLossKernel(args ...tensor.Tensor) error {
	return runMetalPairLossKernel(metalLossHuber, args...)
}

func runMetalBinaryCrossEntropyLossKernel(args ...tensor.Tensor) error {
	return runMetalPairLossKernel(metalLossBinaryCrossEntropy, args...)
}

func runMetalKLDivergenceKernel(args ...tensor.Tensor) error {
	return runMetalPairLossKernel(metalLossKLDivergence, args...)
}

func runMetalCrossEntropyLossKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalCrossEntropyLoss(args[0], args[1], args[2])
}
