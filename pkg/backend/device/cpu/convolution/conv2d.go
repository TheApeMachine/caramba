package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Conv2d — 2D convolution with stride, padding, and dilation. The host
reference is the naive seven-loop implementation; vendor primitives
(cuDNN, MPS-Graph conv) handle the fast paths on devices.

Tensor shapes (NCHW layout):
  - input  [batch, inChannels, inHeight, inWidth]
  - weight [outChannels, inChannels, kernelHeight, kernelWidth]
  - bias   [outChannels]
  - output [batch, outChannels, outHeight, outWidth]

Where:
  outHeight = (inHeight + 2*padH - dilH*(kH-1) - 1) / strideH + 1
  outWidth  = (inWidth  + 2*padW - dilW*(kW-1) - 1) / strideW + 1

Args order for the dispatcher: (input, weight, bias, output). Stride,
padding, and dilation are bound through Conv2DConfig via the typed
Conv2DFloat32 entry point.
*/

/*
Conv2DFloat32 runs Conv2d with the supplied configuration. Shape
checks are exhaustive; the loop body is the naive implementation
suitable as a reference for device-kernel parity.
*/
func Conv2DFloat32(
	config Conv2DConfig,
	input, weight, bias, output tensor.Tensor,
) error {
	inputDims := input.Shape().Dims()
	weightDims := weight.Shape().Dims()
	biasDims := bias.Shape().Dims()
	outputDims := output.Shape().Dims()

	if len(inputDims) != 4 || len(weightDims) != 4 ||
		len(biasDims) != 1 || len(outputDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inputDims[0]
	inChannels := inputDims[1]
	inHeight := inputDims[2]
	inWidth := inputDims[3]

	outChannels := weightDims[0]
	kernelInChannels := weightDims[1]
	kernelHeight := weightDims[2]
	kernelWidth := weightDims[3]

	outHeight := outputDims[2]
	outWidth := outputDims[3]

	if kernelInChannels != inChannels ||
		biasDims[0] != outChannels ||
		outputDims[0] != batch ||
		outputDims[1] != outChannels {
		return tensor.ErrShapeMismatch
	}

	inputView, err := input.Float32Native()

	if err != nil {
		return err
	}

	weightView, err := weight.Float32Native()

	if err != nil {
		return err
	}

	biasView, err := bias.Float32Native()

	if err != nil {
		return err
	}

	outputView, err := output.Float32Native()

	if err != nil {
		return err
	}

	Conv2D(
		config,
		unsafe.Pointer(unsafe.SliceData(inputView)),
		unsafe.Pointer(unsafe.SliceData(weightView)),
		unsafe.Pointer(unsafe.SliceData(biasView)),
		unsafe.Pointer(unsafe.SliceData(outputView)),
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		dtype.Float32,
	)

	return nil
}
