package pooling

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestMaxPool2d_Forward(test *testing.T) {
	Convey("Given a MaxPool2d operation", test, func() {
		operation := NewMaxPool2d(3, 3, 1, 1, 1, 1, 1, 1, false)

		Convey("It should match reference pooling when window tails contain maxima", func() {
			input := poolingSequence(25)
			input[8] = 99
			stateDict := poolingState(input, 1, 1, 5, 5)
			stateDict.KernelH = 3
			stateDict.KernelW = 3
			stateDict.StrideH = 1
			stateDict.StrideW = 1
			stateDict.PadH = 1
			stateDict.PadW = 1
			stateDict.DilationH = 1
			stateDict.DilationW = 1

			outputState, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			So(outputState.Out, ShouldResemble, referenceMaxPool2d(input, 1, 1, 5, 5, stateDict))
		})

		Convey("It should reduce max across SIMD lengths", func() {
			for _, length := range []int{1, 7, 64, 1024, 8192} {
				values := poolingSequence(length)
				values[length-1] = 1_000_000 + float64(length)

				So(kernelMax(values), ShouldEqual, referenceMax(values))
			}
		})
	})
}

func TestAvgPool2d_Forward(test *testing.T) {
	Convey("Given an AvgPool2d operation", test, func() {
		operation := NewAvgPool2d(3, 3, 1, 1, 1, 1, 1, 1, false, false, 0)

		Convey("It should match reference pooling when window tails contribute to the sum", func() {
			input := poolingSequence(25)
			stateDict := poolingState(input, 1, 1, 5, 5)
			stateDict.KernelH = 3
			stateDict.KernelW = 3
			stateDict.StrideH = 1
			stateDict.StrideW = 1
			stateDict.PadH = 1
			stateDict.PadW = 1
			stateDict.DilationH = 1
			stateDict.DilationW = 1

			outputState, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			assertPoolingAlmostEqual(outputState.Out, referenceAvgPool2d(input, 1, 1, 5, 5, stateDict))
		})

		Convey("It should reduce sums across SIMD lengths", func() {
			for _, length := range []int{1, 7, 64, 1024, 8192} {
				values := poolingSequence(length)

				So(kernelSum(values), ShouldEqual, referenceSum(values))
			}
		})
	})
}

func TestAdaptiveAvgPool2d_Forward(test *testing.T) {
	Convey("Given an AdaptiveAvgPool2d operation", test, func() {
		operation := NewAdaptiveAvgPool2d(3, 2)
		input := poolingSequence(30)
		stateDict := poolingState(input, 1, 1, 5, 6)
		stateDict.OutH = 3
		stateDict.OutW = 2

		Convey("It should match reference adaptive average pooling", func() {
			outputState, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			assertPoolingAlmostEqual(
				outputState.Out,
				referenceAdaptiveAvgPool2d(input, 1, 1, 5, 6, 3, 2),
			)
		})
	})
}

func TestAdaptiveMaxPool2d_Forward(test *testing.T) {
	Convey("Given an AdaptiveMaxPool2d operation", test, func() {
		operation := NewAdaptiveMaxPool2d(3, 2)
		input := poolingSequence(30)
		stateDict := poolingState(input, 1, 1, 5, 6)
		stateDict.OutH = 3
		stateDict.OutW = 2

		Convey("It should match reference adaptive max pooling", func() {
			outputState, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			So(outputState.Out, ShouldResemble, referenceAdaptiveMaxPool2d(input, 1, 1, 5, 6, 3, 2))
		})
	})
}

func BenchmarkMaxPool2d_Forward(benchmark *testing.B) {
	operation := NewMaxPool2d(3, 3, 1, 1, 1, 1, 1, 1, false)
	stateDict := poolingState(poolingSequence(4*16*32*32), 4, 16, 32, 32)
	stateDict.KernelH = 3
	stateDict.KernelW = 3
	stateDict.StrideH = 1
	stateDict.StrideW = 1
	stateDict.PadH = 1
	stateDict.PadW = 1
	stateDict.DilationH = 1
	stateDict.DilationW = 1

	for benchmark.Loop() {
		_, _ = operation.Forward(stateDict)
	}
}

func BenchmarkAvgPool2d_Forward(benchmark *testing.B) {
	operation := NewAvgPool2d(3, 3, 1, 1, 1, 1, 1, 1, false, false, 0)
	stateDict := poolingState(poolingSequence(4*16*32*32), 4, 16, 32, 32)
	stateDict.KernelH = 3
	stateDict.KernelW = 3
	stateDict.StrideH = 1
	stateDict.StrideW = 1
	stateDict.PadH = 1
	stateDict.PadW = 1
	stateDict.DilationH = 1
	stateDict.DilationW = 1

	for benchmark.Loop() {
		_, _ = operation.Forward(stateDict)
	}
}

func BenchmarkAdaptiveAvgPool2d_Forward(benchmark *testing.B) {
	operation := NewAdaptiveAvgPool2d(8, 8)
	stateDict := poolingState(poolingSequence(4*16*32*32), 4, 16, 32, 32)
	stateDict.OutH = 8
	stateDict.OutW = 8

	for benchmark.Loop() {
		_, _ = operation.Forward(stateDict)
	}
}

func BenchmarkAdaptiveMaxPool2d_Forward(benchmark *testing.B) {
	operation := NewAdaptiveMaxPool2d(8, 8)
	stateDict := poolingState(poolingSequence(4*16*32*32), 4, 16, 32, 32)
	stateDict.OutH = 8
	stateDict.OutW = 8

	for benchmark.Loop() {
		_, _ = operation.Forward(stateDict)
	}
}

func poolingState(input []float64, batch, channels, height, width int) *state.Dict {
	return state.NewDict().
		WithShape([]int{batch, channels, height, width}).
		WithInput(input)
}

func poolingSequence(length int) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = float64((index*37)%101 - 50)
	}

	return values
}

func referenceMaxPool2d(
	input []float64,
	batch,
	channels,
	height,
	width int,
	stateDict *state.Dict,
) []float64 {
	outputHeight := outSizeMax(
		height, stateDict.KernelH, stateDict.StrideH,
		stateDict.PadH, stateDict.DilationH, stateDict.Ceil,
	)
	outputWidth := outSizeMax(
		width, stateDict.KernelW, stateDict.StrideW,
		stateDict.PadW, stateDict.DilationW, stateDict.Ceil,
	)
	output := make([]float64, batch*channels*outputHeight*outputWidth)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			referencePool2dWindow(
				input, output, batchIndex, channelIndex,
				channels, height, width, outputHeight, outputWidth, stateDict, true,
			)
		}
	}

	return output
}

func referenceAvgPool2d(
	input []float64,
	batch,
	channels,
	height,
	width int,
	stateDict *state.Dict,
) []float64 {
	outputHeight := outSizeMax(
		height, stateDict.KernelH, stateDict.StrideH,
		stateDict.PadH, stateDict.DilationH, stateDict.Ceil,
	)
	outputWidth := outSizeMax(
		width, stateDict.KernelW, stateDict.StrideW,
		stateDict.PadW, stateDict.DilationW, stateDict.Ceil,
	)
	output := make([]float64, batch*channels*outputHeight*outputWidth)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			referencePool2dWindow(
				input, output, batchIndex, channelIndex,
				channels, height, width, outputHeight, outputWidth, stateDict, false,
			)
		}
	}

	return output
}

func referencePool2dWindow(
	input,
	output []float64,
	batchIndex,
	channelIndex,
	channels,
	height,
	width,
	outputHeight,
	outputWidth int,
	stateDict *state.Dict,
	maxPool bool,
) {
	baseInput := (batchIndex*channels + channelIndex) * height * width
	baseOutput := (batchIndex*channels + channelIndex) * outputHeight * outputWidth

	for outputH := 0; outputH < outputHeight; outputH++ {
		for outputW := 0; outputW < outputWidth; outputW++ {
			values, kernelCount := referencePoolWindowValues(
				input, baseInput, height, width, outputH, outputW, stateDict,
			)

			if maxPool {
				output[baseOutput+outputH*outputWidth+outputW] = referenceMax(values)
				continue
			}

			divisor := len(values)

			if stateDict.Divisor != 0 {
				divisor = stateDict.Divisor
			}

			if stateDict.CountPad {
				divisor = kernelCount
			}

			output[baseOutput+outputH*outputWidth+outputW] = referenceSum(values) / float64(divisor)
		}
	}
}

func referencePoolWindowValues(
	input []float64,
	baseInput,
	height,
	width,
	outputH,
	outputW int,
	stateDict *state.Dict,
) ([]float64, int) {
	values := make([]float64, 0, stateDict.KernelH*stateDict.KernelW)
	kernelCount := 0
	heightStart := outputH*stateDict.StrideH - stateDict.PadH
	widthStart := outputW*stateDict.StrideW - stateDict.PadW

	for kernelH := 0; kernelH < stateDict.KernelH; kernelH++ {
		inputH := heightStart + kernelH*stateDict.DilationH

		for kernelW := 0; kernelW < stateDict.KernelW; kernelW++ {
			kernelCount++
			inputW := widthStart + kernelW*stateDict.DilationW

			if inputH < 0 || inputH >= height || inputW < 0 || inputW >= width {
				continue
			}

			values = append(values, input[baseInput+inputH*width+inputW])
		}
	}

	return values, kernelCount
}

func referenceAdaptiveAvgPool2d(
	input []float64,
	batch,
	channels,
	height,
	width,
	outputHeight,
	outputWidth int,
) []float64 {
	output := make([]float64, batch*channels*outputHeight*outputWidth)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			referenceAdaptivePool2dWindow(
				input, output, batchIndex, channelIndex,
				channels, height, width, outputHeight, outputWidth, false,
			)
		}
	}

	return output
}

func referenceAdaptiveMaxPool2d(
	input []float64,
	batch,
	channels,
	height,
	width,
	outputHeight,
	outputWidth int,
) []float64 {
	output := make([]float64, batch*channels*outputHeight*outputWidth)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			referenceAdaptivePool2dWindow(
				input, output, batchIndex, channelIndex,
				channels, height, width, outputHeight, outputWidth, true,
			)
		}
	}

	return output
}

func referenceAdaptivePool2dWindow(
	input,
	output []float64,
	batchIndex,
	channelIndex,
	channels,
	height,
	width,
	outputHeight,
	outputWidth int,
	maxPool bool,
) {
	baseInput := (batchIndex*channels + channelIndex) * height * width
	baseOutput := (batchIndex*channels + channelIndex) * outputHeight * outputWidth

	for outputH := 0; outputH < outputHeight; outputH++ {
		heightStart := outputH * height / outputHeight
		heightEnd := ceilDiv(outputH+1, outputHeight, height)

		for outputW := 0; outputW < outputWidth; outputW++ {
			widthStart := outputW * width / outputWidth
			widthEnd := ceilDiv(outputW+1, outputWidth, width)
			sum := 0.0
			maxValue := math.Inf(-1)
			count := 0

			for inputH := heightStart; inputH < heightEnd; inputH++ {
				for inputW := widthStart; inputW < widthEnd; inputW++ {
					value := input[baseInput+inputH*width+inputW]
					sum += value
					count++

					if value > maxValue {
						maxValue = value
					}
				}
			}

			if maxPool {
				output[baseOutput+outputH*outputWidth+outputW] = maxValue
				continue
			}

			output[baseOutput+outputH*outputWidth+outputW] = sum / float64(count)
		}
	}
}

func referenceMax(values []float64) float64 {
	maxValue := math.Inf(-1)

	for _, value := range values {
		if value > maxValue {
			maxValue = value
		}
	}

	return maxValue
}

func referenceSum(values []float64) float64 {
	var sum float64

	for _, value := range values {
		sum += value
	}

	return sum
}

func assertPoolingAlmostEqual(actual, expected []float64) {
	So(len(actual), ShouldEqual, len(expected))

	for index := range expected {
		So(actual[index], ShouldAlmostEqual, expected[index], 1e-12)
	}
}
