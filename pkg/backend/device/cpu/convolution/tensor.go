package convolution

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func RunConv2DDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return Conv2DFloat32(
		DefaultConv2DConfig(),
		args[0], args[1], args[2], args[3],
	)
}

func RunConv1DDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return Conv1DFloat32(DefaultConv1DConfig(), args[0], args[1], args[2], args[3])
}

func RunConv3DDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return Conv3DFloat32(DefaultConv3DConfig(), args[0], args[1], args[2], args[3])
}

func RunConvTranspose2DDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return ConvTranspose2DFloat32(DefaultConv2DConfig(), args[0], args[1], args[2], args[3])
}
