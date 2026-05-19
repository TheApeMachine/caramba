package pool

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func RunMaxPool2DDefault(args ...tensor.Tensor) error {
	return MaxPool2DFloat32(DefaultPoolConfig(), args[0], args[1])
}

func RunAvgPool2DDefault(args ...tensor.Tensor) error {
	return AvgPool2DFloat32(DefaultPoolConfig(), args[0], args[1])
}

func MaxPool2DFloat32(config PoolConfig, input, out tensor.Tensor) error {
	return pool2DFloat32(config, input, out, true)
}

func AvgPool2DFloat32(config PoolConfig, input, out tensor.Tensor) error {
	return pool2DFloat32(config, input, out, false)
}

func RunAdaptiveAvgPool2DDefault(args ...tensor.Tensor) error {
	return AdaptiveAvgPool2DFloat32(args[0], args[1])
}

func RunAdaptiveMaxPool2DDefault(args ...tensor.Tensor) error {
	return AdaptiveMaxPool2DFloat32(args[0], args[1])
}

func AdaptiveAvgPool2DFloat32(input, out tensor.Tensor) error {
	return adaptivePool2DFloat32(input, out, false)
}

func AdaptiveMaxPool2DFloat32(input, out tensor.Tensor) error {
	return adaptivePool2DFloat32(input, out, true)
}

func adaptivePool2DFloat32(input, out tensor.Tensor, useMax bool) error {
	inputDims := input.Shape().Dims()
	outDims := out.Shape().Dims()

	if len(inputDims) != 4 || len(outDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inputDims[0]
	channels := inputDims[1]
	inHeight := inputDims[2]
	inWidth := inputDims[3]
	outHeight := outDims[2]
	outWidth := outDims[3]

	if outDims[0] != batch || outDims[1] != channels {
		return tensor.ErrShapeMismatch
	}

	inputView, err := input.Float32Native()

	if err != nil {
		return err
	}

	outputView, err := out.Float32Native()

	if err != nil {
		return err
	}

	if len(inputView) == 0 {
		return nil
	}

	if useMax {
		AdaptiveMaxPool2D(
			unsafe.Pointer(&inputView[0]),
			unsafe.Pointer(&outputView[0]),
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			dtype.Float32,
		)

		return nil
	}

	AdaptiveAvgPool2D(
		unsafe.Pointer(&inputView[0]),
		unsafe.Pointer(&outputView[0]),
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		dtype.Float32,
	)

	return nil
}

func pool2DFloat32(
	config PoolConfig,
	input, out tensor.Tensor,
	useMax bool,
) error {
	inputDims := input.Shape().Dims()
	outDims := out.Shape().Dims()

	if len(inputDims) != 4 || len(outDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inputDims[0]
	channels := inputDims[1]
	inHeight := inputDims[2]
	inWidth := inputDims[3]

	outHeight := outDims[2]
	outWidth := outDims[3]

	if outDims[0] != batch || outDims[1] != channels {
		return tensor.ErrShapeMismatch
	}

	inputView, err := input.Float32Native()

	if err != nil {
		return err
	}

	outputView, err := out.Float32Native()

	if err != nil {
		return err
	}

	if len(inputView) == 0 {
		return nil
	}

	if useMax {
		MaxPool2D(
			config,
			unsafe.Pointer(&inputView[0]),
			unsafe.Pointer(&outputView[0]),
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			dtype.Float32,
		)

		return nil
	}

	AvgPool2D(
		config,
		unsafe.Pointer(&inputView[0]),
		unsafe.Pointer(&outputView[0]),
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		dtype.Float32,
	)

	return nil
}
