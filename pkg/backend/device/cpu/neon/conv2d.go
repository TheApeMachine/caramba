package neon

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
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

type Conv2DConfig struct {
	StrideH   int
	StrideW   int
	PaddingH  int
	PaddingW  int
	DilationH int
	DilationW int
}

func DefaultConv2DConfig() Conv2DConfig {
	return Conv2DConfig{
		StrideH: 1, StrideW: 1,
		PaddingH: 0, PaddingW: 0,
		DilationH: 1, DilationW: 1,
	}
}

func init() {
	Default.Register(Kernel{
		Name: "conv2d",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runConv2DDefault,
	})
}

func runConv2DDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return Conv2DFloat32(
		DefaultConv2DConfig(),
		args[0], args[1], args[2], args[3],
	)
}

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

	Conv2DFloat32Native(
		config,
		inputView, weightView, biasView, outputView,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)

	return nil
}

func conv2DFloat32Scalar(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	for batchIndex := range batch {
		for outChIndex := range outChannels {
			for outRow := range outHeight {
				for outCol := range outWidth {
					outIdx := ((batchIndex*outChannels+outChIndex)*outHeight+outRow)*outWidth + outCol
					outputView[outIdx] = Conv2DPixelScalar(
						config,
						inputView, weightView,
						batchIndex*inChannels*inHeight*inWidth,
						outChIndex*inChannels*kernelHeight*kernelWidth,
						inChannels, inHeight, inWidth,
						kernelHeight, kernelWidth,
						outRow, outCol,
						biasView[outChIndex],
					)
				}
			}
		}
	}
}

func Conv2DPixelScalar(
	config Conv2DConfig,
	inputView, weightView []float32,
	inputBatchOffset, weightChannelOffset int,
	inChannels, inHeight, inWidth,
	kernelHeight, kernelWidth,
	outRow, outCol int,
	biasValue float32,
) float32 {
	sum := biasValue

	for inChIndex := range inChannels {
		for kRow := range kernelHeight {
			inRow := outRow*config.StrideH + kRow*config.DilationH - config.PaddingH

			if inRow < 0 || inRow >= inHeight {
				continue
			}

			for kCol := range kernelWidth {
				inCol := outCol*config.StrideW + kCol*config.DilationW - config.PaddingW

				if inCol < 0 || inCol >= inWidth {
					continue
				}

				inputIdx := inputBatchOffset + (inChIndex*inHeight+inRow)*inWidth + inCol
				weightIdx := weightChannelOffset + (inChIndex*kernelHeight+kRow)*kernelWidth + kCol

				sum += inputView[inputIdx] * weightView[weightIdx]
			}
		}
	}

	return sum
}
