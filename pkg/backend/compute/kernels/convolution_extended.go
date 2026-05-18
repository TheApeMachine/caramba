package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
1-D and 3-D convolutions plus 2-D transposed convolution. Host
references follow the standard NCL / NCDHW layout conventions.
*/

type Conv1DConfig struct {
	Stride   int
	Padding  int
	Dilation int
}

func DefaultConv1DConfig() Conv1DConfig {
	return Conv1DConfig{Stride: 1, Padding: 0, Dilation: 1}
}

type Conv3DConfig struct {
	StrideD, StrideH, StrideW         int
	PaddingD, PaddingH, PaddingW      int
	DilationD, DilationH, DilationW   int
}

func DefaultConv3DConfig() Conv3DConfig {
	return Conv3DConfig{
		StrideD: 1, StrideH: 1, StrideW: 1,
		DilationD: 1, DilationH: 1, DilationW: 1,
	}
}

func init() {
	Default.Register(Kernel{
		Name: "conv1d",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runConv1DDefault,
	})

	Default.Register(Kernel{
		Name: "conv3d",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runConv3DDefault,
	})

	Default.Register(Kernel{
		Name: "conv_transpose2d",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runConvTranspose2DDefault,
	})

	Default.Register(Kernel{
		Name: "adaptive_avg_pool2d",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAdaptiveAvgPool2D,
	})

	Default.Register(Kernel{
		Name: "adaptive_max_pool2d",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAdaptiveMaxPool2D,
	})
}

func runConv1DDefault(args ...tensor.Tensor) error {
	return Conv1DFloat32(DefaultConv1DConfig(), args[0], args[1], args[2], args[3])
}

func runConv3DDefault(args ...tensor.Tensor) error {
	return Conv3DFloat32(DefaultConv3DConfig(), args[0], args[1], args[2], args[3])
}

func runConvTranspose2DDefault(args ...tensor.Tensor) error {
	return ConvTranspose2DFloat32(DefaultConv2DConfig(), args[0], args[1], args[2], args[3])
}

/*
Conv1DFloat32 — 1-D convolution. Shapes:
  - input  [batch, inChannels, inLength]
  - weight [outChannels, inChannels, kernelLength]
  - bias   [outChannels]
  - output [batch, outChannels, outLength]
*/
func Conv1DFloat32(config Conv1DConfig, input, weight, bias, output tensor.Tensor) error {
	inputView, _ := input.Float32Native()
	weightView, _ := weight.Float32Native()
	biasView, _ := bias.Float32Native()
	outputView, _ := output.Float32Native()

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

	conv1DFloat32Native(
		config,
		inputView, weightView, biasView, outputView,
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
	)

	return nil
}

/*
Conv3DFloat32 — 3-D convolution. Naive seven-loop reference for
shape parity. Shapes follow NCDHW.
*/
func Conv3DFloat32(config Conv3DConfig, input, weight, bias, output tensor.Tensor) error {
	inputView, _ := input.Float32Native()
	weightView, _ := weight.Float32Native()
	biasView, _ := bias.Float32Native()
	outputView, _ := output.Float32Native()

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

	conv3DFloat32Native(
		config,
		inputView, weightView, biasView, outputView,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW,
	)

	return nil
}

/*
ConvTranspose2DFloat32 — 2-D transposed convolution (deconv). Used
by generative diffusion models and U-Nets. Implemented as the
gradient of conv2d w.r.t. its input.
*/
func ConvTranspose2DFloat32(config Conv2DConfig, input, weight, bias, output tensor.Tensor) error {
	inputView, _ := input.Float32Native()
	weightView, _ := weight.Float32Native()
	biasView, _ := bias.Float32Native()
	outputView, _ := output.Float32Native()

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

	// Initialize output with bias.
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for outChIndex := 0; outChIndex < outChannels; outChIndex++ {
			for outRow := 0; outRow < outHeight; outRow++ {
				for outCol := 0; outCol < outWidth; outCol++ {
					idx := ((batchIndex*outChannels+outChIndex)*outHeight+outRow)*outWidth + outCol
					outputView[idx] = biasView[outChIndex]
				}
			}
		}
	}

	// Transposed conv: scatter input values into output via weights.
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for inChIndex := 0; inChIndex < inChannels; inChIndex++ {
			for inRow := 0; inRow < inHeight; inRow++ {
				for inCol := 0; inCol < inWidth; inCol++ {
					inputValue := inputView[((batchIndex*inChannels+inChIndex)*inHeight+inRow)*inWidth+inCol]

					for outChIndex := 0; outChIndex < outChannels; outChIndex++ {
						for kRow := 0; kRow < kernelHeight; kRow++ {
							for kCol := 0; kCol < kernelWidth; kCol++ {
								outRow := inRow*config.StrideH + kRow*config.DilationH - config.PaddingH
								outCol := inCol*config.StrideW + kCol*config.DilationW - config.PaddingW

								if outRow < 0 || outRow >= outHeight || outCol < 0 || outCol >= outWidth {
									continue
								}

								weightIdx := ((inChIndex*outChannels+outChIndex)*kernelHeight+kRow)*kernelWidth + kCol
								outIdx := ((batchIndex*outChannels+outChIndex)*outHeight+outRow)*outWidth + outCol

								outputView[outIdx] += inputValue * weightView[weightIdx]
							}
						}
					}
				}
			}
		}
	}

	return nil
}

/*
runAdaptiveAvgPool2D pools to a fixed output spatial size by
dividing the input region per output cell.
*/
func runAdaptiveAvgPool2D(args ...tensor.Tensor) error {
	return adaptivePool2D(args, false)
}

func runAdaptiveMaxPool2D(args ...tensor.Tensor) error {
	return adaptivePool2D(args, true)
}

func adaptivePool2D(args []tensor.Tensor, useMax bool) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inputView, _ := args[0].Float32Native()
	outputView, _ := args[1].Float32Native()

	inDims := args[0].Shape().Dims()
	outDims := args[1].Shape().Dims()

	if len(inDims) != 4 || len(outDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	channels := inDims[1]
	inH := inDims[2]
	inW := inDims[3]
	outH := outDims[2]
	outW := outDims[3]

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for chIndex := 0; chIndex < channels; chIndex++ {
			for outRow := 0; outRow < outH; outRow++ {
				startRow := (outRow * inH) / outH
				endRow := ((outRow + 1) * inH) / outH

				for outCol := 0; outCol < outW; outCol++ {
					startCol := (outCol * inW) / outW
					endCol := ((outCol + 1) * inW) / outW

					value := outputAdaptivePoolValue(
						inputView, batchIndex, chIndex, channels,
						inH, inW, startRow, endRow, startCol, endCol, useMax,
					)

					outputView[((batchIndex*channels+chIndex)*outH+outRow)*outW+outCol] = value
				}
			}
		}
	}

	return nil
}

func outputAdaptivePoolValue(
	inputView []float32,
	batchIndex, chIndex, channels, inH, inW int,
	startRow, endRow, startCol, endCol int,
	useMax bool,
) float32 {
	var sum float32
	maximum := float32(-1e30)
	count := 0

	for row := startRow; row < endRow; row++ {
		for col := startCol; col < endCol; col++ {
			value := inputView[((batchIndex*channels+chIndex)*inH+row)*inW+col]
			count++

			if useMax {
				if value > maximum {
					maximum = value
				}

				continue
			}

			sum += value
		}
	}

	if useMax {
		return maximum
	}

	if count == 0 {
		return 0
	}

	return sum / float32(count)
}
