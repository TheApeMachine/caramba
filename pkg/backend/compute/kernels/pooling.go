package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Pooling kernels for 2-D inputs. Args order: (input, output) with
config carrying kernel/stride/padding. Both max and average variants
land here.
*/

type PoolConfig struct {
	KernelH  int
	KernelW  int
	StrideH  int
	StrideW  int
	PaddingH int
	PaddingW int
}

func DefaultPoolConfig() PoolConfig {
	return PoolConfig{KernelH: 2, KernelW: 2, StrideH: 2, StrideW: 2}
}

func init() {
	Default.Register(Kernel{
		Name: "max_pool2d",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMaxPool2DDefault,
	})

	Default.Register(Kernel{
		Name: "avg_pool2d",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAvgPool2DDefault,
	})
}

func runMaxPool2DDefault(args ...tensor.Tensor) error {
	return MaxPool2DFloat32(DefaultPoolConfig(), args[0], args[1])
}

func runAvgPool2DDefault(args ...tensor.Tensor) error {
	return AvgPool2DFloat32(DefaultPoolConfig(), args[0], args[1])
}

/*
MaxPool2DFloat32 operates on [batch, channels, height, width] NCHW
tensors. Output dimensions follow the standard formula.
*/
func MaxPool2DFloat32(config PoolConfig, input, out tensor.Tensor) error {
	return pool2DFloat32(config, input, out, true)
}

/*
AvgPool2DFloat32 mirrors MaxPool2DFloat32 with arithmetic mean.
*/
func AvgPool2DFloat32(config PoolConfig, input, out tensor.Tensor) error {
	return pool2DFloat32(config, input, out, false)
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

	inputView, _ := input.Float32Native()
	outputView, _ := out.Float32Native()

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth

			for outRow := 0; outRow < outHeight; outRow++ {
				for outCol := 0; outCol < outWidth; outCol++ {
					value := poolWindow(
						inputView[channelOffsetIn:channelOffsetIn+inHeight*inWidth],
						inHeight, inWidth,
						outRow, outCol,
						config, useMax,
					)

					outputView[channelOffsetOut+outRow*outWidth+outCol] = value
				}
			}
		}
	}

	return nil
}

func poolWindow(
	channel []float32,
	inHeight, inWidth int,
	outRow, outCol int,
	config PoolConfig,
	useMax bool,
) float32 {
	startRow := outRow*config.StrideH - config.PaddingH
	startCol := outCol*config.StrideW - config.PaddingW

	value := float32(math.Inf(-1))

	if !useMax {
		value = 0
	}

	count := 0

	for kernelRow := 0; kernelRow < config.KernelH; kernelRow++ {
		for kernelCol := 0; kernelCol < config.KernelW; kernelCol++ {
			row := startRow + kernelRow
			col := startCol + kernelCol

			if row < 0 || row >= inHeight || col < 0 || col >= inWidth {
				continue
			}

			candidate := channel[row*inWidth+col]
			count++

			switch {
			case useMax:
				if candidate > value {
					value = candidate
				}
			default:
				value += candidate
			}
		}
	}

	if !useMax && count > 0 {
		value /= float32(count)
	}

	return value
}
