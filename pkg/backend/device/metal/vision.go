package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalVisionDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalVisionDTypes {
		registerMetalVisionKernels(storageDType)
	}
}

func registerMetalVisionKernels(storageDType dtype.DType) {
	registerMetalConv1DKernel(storageDType)
	registerMetalConv2DKernel(storageDType)
	registerMetalConv3DKernel(storageDType)
	registerMetalConvTranspose2DKernel(storageDType)
	registerMetalPool2DKernel("max_pool2d", storageDType, runMetalMaxPool2DKernel)
	registerMetalPool2DKernel("avg_pool2d", storageDType, runMetalAvgPool2DKernel)
	registerMetalPool2DKernel("adaptive_avg_pool2d", storageDType, runMetalAdaptiveAvgPool2DKernel)
	registerMetalPool2DKernel("adaptive_max_pool2d", storageDType, runMetalAdaptiveMaxPool2DKernel)
}

func registerMetalConv1DKernel(storageDType dtype.DType) {
	registerMetalConvolutionKernel("conv1d", storageDType, runMetalConv1DKernel)
}

func registerMetalConv2DKernel(storageDType dtype.DType) {
	registerMetalConvolutionKernel("conv2d", storageDType, runMetalConv2DKernel)
}

func registerMetalConv3DKernel(storageDType dtype.DType) {
	registerMetalConvolutionKernel("conv3d", storageDType, runMetalConv3DKernel)
}

func registerMetalConvTranspose2DKernel(storageDType dtype.DType) {
	registerMetalConvolutionKernel("conv_transpose2d", storageDType, runMetalConvTranspose2DKernel)
}

func registerMetalConvolutionKernel(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
	})
}

func registerMetalPool2DKernel(
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

func runMetalConv1DKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalConv1D(args[0], args[1], args[2], args[3])
}

func runMetalConv2DKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalConv2D(args[0], args[1], args[2], args[3])
}

func runMetalConv3DKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalConv3D(args[0], args[1], args[2], args[3])
}

func runMetalConvTranspose2DKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalConvTranspose2D(args[0], args[1], args[2], args[3])
}

func runMetalMaxPool2DKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalMaxPool2D(args[0], args[1])
}

func runMetalAvgPool2DKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalAvgPool2D(args[0], args[1])
}

func runMetalAdaptiveAvgPool2DKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalAdaptiveAvgPool2D(args[0], args[1])
}

func runMetalAdaptiveMaxPool2DKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalAdaptiveMaxPool2D(args[0], args[1])
}
