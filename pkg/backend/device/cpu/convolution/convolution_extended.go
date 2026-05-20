package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
1-D and 3-D convolutions plus 2-D transposed convolution. Host
references follow the standard NCL / NCDHW layout conventions.
*/

/*
Conv1DFloat32 — 1-D convolution. Shapes:
  - input  [batch, inChannels, inLength]
  - weight [outChannels, inChannels, kernelLength]
  - bias   [outChannels]
  - output [batch, outChannels, outLength]
*/
func Conv1DFloat32(config Conv1DConfig, input, weight, bias, output tensor.Tensor) error {
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

	inDims := input.Shape().Dims()
	wDims := weight.Shape().Dims()
	outDims := output.Shape().Dims()

	if len(inDims) != 3 || len(wDims) != 3 || len(outDims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	inChannels := inDims[1]
	inLength := inDims[2]
	outChannels := wDims[0]
	kernelLength := wDims[2]
	outLength := outDims[2]

	Conv1D(
		config,
		unsafe.Pointer(unsafe.SliceData(inputView)),
		unsafe.Pointer(unsafe.SliceData(weightView)),
		unsafe.Pointer(unsafe.SliceData(biasView)),
		unsafe.Pointer(unsafe.SliceData(outputView)),
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
		dtype.Float32,
	)

	return nil
}

/*
Conv3DFloat32 — 3-D convolution. Naive seven-loop reference for
shape parity. Shapes follow NCDHW.
*/
func Conv3DFloat32(config Conv3DConfig, input, weight, bias, output tensor.Tensor) error {
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

	inDims := input.Shape().Dims()
	wDims := weight.Shape().Dims()
	outDims := output.Shape().Dims()

	if len(inDims) != 5 || len(wDims) != 5 || len(outDims) != 5 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	inChannels := inDims[1]
	inD := inDims[2]
	inH := inDims[3]
	inW := inDims[4]
	outChannels := wDims[0]
	kD := wDims[2]
	kH := wDims[3]
	kW := wDims[4]
	outD := outDims[2]
	outH := outDims[3]
	outW := outDims[4]

	Conv3D(
		config,
		unsafe.Pointer(unsafe.SliceData(inputView)),
		unsafe.Pointer(unsafe.SliceData(weightView)),
		unsafe.Pointer(unsafe.SliceData(biasView)),
		unsafe.Pointer(unsafe.SliceData(outputView)),
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW,
		dtype.Float32,
	)

	return nil
}

/*
ConvTranspose2DFloat32 — 2-D transposed convolution (deconv). Used
by generative diffusion models and U-Nets. Implemented as the
gradient of conv2d w.r.t. its input.
*/
func ConvTranspose2DFloat32(config Conv2DConfig, input, weight, bias, output tensor.Tensor) error {
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

	inDims := input.Shape().Dims()
	wDims := weight.Shape().Dims()
	outDims := output.Shape().Dims()

	if len(inDims) != 4 || len(wDims) != 4 || len(outDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	inChannels := inDims[1]
	inHeight := inDims[2]
	inWidth := inDims[3]
	outChannels := wDims[1]
	kernelHeight := wDims[2]
	kernelWidth := wDims[3]
	outHeight := outDims[2]
	outWidth := outDims[3]

	ConvTranspose2D(
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
