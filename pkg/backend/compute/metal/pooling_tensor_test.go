//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestPoolingOps_MaxPool2dTensor(test *testing.T) {
	Convey("Given resident Metal max pooling inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		poolingOps := metalPoolingOpsForTest(test, tensorBackend)
		shape, outputShape, input := pool2dCase(test)
		params := maxPoolTestParams()
		expected := referencePool2d(input, shape.Dims(), outputShape.Dims(), params, true)

		Convey("It should match the scalar NCHW reference", func() {
			output := runPool2dTensorForTest(
				test,
				tensorBackend,
				shape,
				outputShape,
				input,
				func(inputTensor computetensor.Tensor) (computetensor.Tensor, error) {
					return poolingOps.MaxPool2dTensor(inputTensor, outputShape, params)
				},
			)

			So(output.Location(), ShouldEqual, computetensor.Metal)
			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()
			assertMetalMaxDiff(values, expected, 1e-6)
		})
	})
}

func TestPoolingOps_AvgPool2dTensor(test *testing.T) {
	Convey("Given resident Metal average pooling inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		poolingOps := metalPoolingOpsForTest(test, tensorBackend)
		shape, outputShape, input := pool2dCase(test)
		params := avgPoolTestParams()
		expected := referencePool2d(input, shape.Dims(), outputShape.Dims(), maxParamsFromAvg(params), false)

		Convey("It should match the scalar NCHW reference", func() {
			output := runPool2dTensorForTest(
				test,
				tensorBackend,
				shape,
				outputShape,
				input,
				func(inputTensor computetensor.Tensor) (computetensor.Tensor, error) {
					return poolingOps.AvgPool2dTensor(inputTensor, outputShape, params)
				},
			)

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()
			assertMetalMaxDiff(values, expected, 1e-6)
		})
	})
}

func TestPoolingOps_AdaptiveAvgPool2dTensor(test *testing.T) {
	Convey("Given resident Metal adaptive average pooling inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		poolingOps := metalPoolingOpsForTest(test, tensorBackend)
		shape, outputShape, input := adaptivePool2dCase(test)
		expected := referenceAdaptivePool2d(input, shape.Dims(), outputShape.Dims(), false)

		Convey("It should match the scalar adaptive reference", func() {
			output := runPool2dTensorForTest(
				test,
				tensorBackend,
				shape,
				outputShape,
				input,
				func(inputTensor computetensor.Tensor) (computetensor.Tensor, error) {
					return poolingOps.AdaptiveAvgPool2dTensor(inputTensor, outputShape)
				},
			)

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()
			assertMetalMaxDiff(values, expected, 1e-6)
		})
	})
}

func TestPoolingOps_AdaptiveMaxPool2dTensor(test *testing.T) {
	Convey("Given resident Metal adaptive max pooling inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		poolingOps := metalPoolingOpsForTest(test, tensorBackend)
		shape, outputShape, input := adaptivePool2dCase(test)
		expected := referenceAdaptivePool2d(input, shape.Dims(), outputShape.Dims(), true)

		Convey("It should match the scalar adaptive reference", func() {
			output := runPool2dTensorForTest(
				test,
				tensorBackend,
				shape,
				outputShape,
				input,
				func(inputTensor computetensor.Tensor) (computetensor.Tensor, error) {
					return poolingOps.AdaptiveMaxPool2dTensor(inputTensor, outputShape)
				},
			)

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()
			So(values, ShouldResemble, expected)
		})
	})
}

type poolTensorOperation func(computetensor.Tensor) (computetensor.Tensor, error)

func metalPoolingOpsForTest(test testing.TB, tensorBackend *TensorBackend) *PoolingOps {
	test.Helper()

	poolingOps, err := tensorBackend.pooling()
	So(err, ShouldBeNil)

	return poolingOps
}

func runPool2dTensorForTest(
	test testing.TB,
	tensorBackend *TensorBackend,
	shape computetensor.Shape,
	outputShape computetensor.Shape,
	input []float64,
	operation poolTensorOperation,
) computetensor.Tensor {
	test.Helper()
	So(outputShape.Valid(), ShouldBeTrue)

	inputTensor := uploadMetalTensorForTest(test, tensorBackend, shape, input)
	output, err := operation(inputTensor)
	So(err, ShouldBeNil)

	return output
}

func pool2dCase(test testing.TB) (computetensor.Shape, computetensor.Shape, []float64) {
	test.Helper()

	shape, err := computetensor.NewShape([]int{1, 1, 5, 5})
	So(err, ShouldBeNil)

	outputShape, err := computetensor.NewShape([]int{1, 1, 5, 5})
	So(err, ShouldBeNil)

	return shape, outputShape, poolingSequence(shape.Len())
}

func adaptivePool2dCase(test testing.TB) (computetensor.Shape, computetensor.Shape, []float64) {
	test.Helper()

	shape, err := computetensor.NewShape([]int{1, 1, 5, 6})
	So(err, ShouldBeNil)

	outputShape, err := computetensor.NewShape([]int{1, 1, 3, 2})
	So(err, ShouldBeNil)

	return shape, outputShape, poolingSequence(shape.Len())
}

func maxPoolTestParams() MaxPool2dParams {
	return MaxPool2dParams{
		KernelH: 3, KernelW: 3,
		StrideH: 1, StrideW: 1,
		PadH: 1, PadW: 1,
		DilationH: 1, DilationW: 1,
	}
}

func avgPoolTestParams() AvgPool2dParams {
	params := maxPoolTestParams()

	return AvgPool2dParams{
		KernelH: params.KernelH, KernelW: params.KernelW,
		StrideH: params.StrideH, StrideW: params.StrideW,
		PadH: params.PadH, PadW: params.PadW,
		DilationH: params.DilationH, DilationW: params.DilationW,
	}
}

func poolingSequence(length int) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = float64((index*37)%101 - 50)
	}

	return values
}

func referencePool2d(
	input []float64,
	inputDims []int,
	outputDims []int,
	params MaxPool2dParams,
	maxPool bool,
) []float64 {
	output := make([]float64, outputDims[0]*outputDims[1]*outputDims[2]*outputDims[3])

	for batchIndex := range inputDims[0] {
		for channelIndex := range inputDims[1] {
			referencePool2dChannel(input, output, inputDims, outputDims, params, maxPool, batchIndex, channelIndex)
		}
	}

	return output
}

func referencePool2dChannel(
	input []float64,
	output []float64,
	inputDims []int,
	outputDims []int,
	params MaxPool2dParams,
	maxPool bool,
	batchIndex int,
	channelIndex int,
) {
	inputBase := (batchIndex*inputDims[1] + channelIndex) * inputDims[2] * inputDims[3]
	outputBase := (batchIndex*outputDims[1] + channelIndex) * outputDims[2] * outputDims[3]

	for outputH := range outputDims[2] {
		for outputW := range outputDims[3] {
			values := referencePoolWindow(input, inputBase, inputDims, params, outputH, outputW)
			output[outputBase+outputH*outputDims[3]+outputW] = referencePoolValues(values, maxPool)
		}
	}
}

func referencePoolWindow(
	input []float64,
	inputBase int,
	inputDims []int,
	params MaxPool2dParams,
	outputH int,
	outputW int,
) []float64 {
	values := make([]float64, 0, params.KernelH*params.KernelW)
	heightStart := outputH*params.StrideH - params.PadH
	widthStart := outputW*params.StrideW - params.PadW

	for kernelH := range params.KernelH {
		inputH := heightStart + kernelH*params.DilationH

		for kernelW := range params.KernelW {
			inputW := widthStart + kernelW*params.DilationW

			if inputH < 0 || inputH >= inputDims[2] || inputW < 0 || inputW >= inputDims[3] {
				continue
			}

			values = append(values, input[inputBase+inputH*inputDims[3]+inputW])
		}
	}

	return values
}

func referencePoolValues(values []float64, maxPool bool) float64 {
	if maxPool {
		return referenceMax(values)
	}

	return referenceSum(values) / float64(len(values))
}

func referenceAdaptivePool2d(input []float64, inputDims []int, outputDims []int, maxPool bool) []float64 {
	output := make([]float64, outputDims[0]*outputDims[1]*outputDims[2]*outputDims[3])

	for batchIndex := range inputDims[0] {
		for channelIndex := range inputDims[1] {
			referenceAdaptivePoolChannel(input, output, inputDims, outputDims, maxPool, batchIndex, channelIndex)
		}
	}

	return output
}

func referenceAdaptivePoolChannel(
	input []float64,
	output []float64,
	inputDims []int,
	outputDims []int,
	maxPool bool,
	batchIndex int,
	channelIndex int,
) {
	inputBase := (batchIndex*inputDims[1] + channelIndex) * inputDims[2] * inputDims[3]
	outputBase := (batchIndex*outputDims[1] + channelIndex) * outputDims[2] * outputDims[3]

	for outputH := range outputDims[2] {
		heightStart := outputH * inputDims[2] / outputDims[2]
		heightEnd := ((outputH+1)*inputDims[2] + outputDims[2] - 1) / outputDims[2]

		for outputW := range outputDims[3] {
			widthStart := outputW * inputDims[3] / outputDims[3]
			widthEnd := ((outputW+1)*inputDims[3] + outputDims[3] - 1) / outputDims[3]
			output[outputBase+outputH*outputDims[3]+outputW] = referenceAdaptiveWindow(
				input[inputBase:], inputDims[3], heightStart, heightEnd, widthStart, widthEnd, maxPool,
			)
		}
	}
}

func referenceAdaptiveWindow(
	input []float64,
	width int,
	heightStart int,
	heightEnd int,
	widthStart int,
	widthEnd int,
	maxPool bool,
) float64 {
	sum := 0.0
	maxValue := math.Inf(-1)
	count := 0

	for inputH := heightStart; inputH < heightEnd; inputH++ {
		for inputW := widthStart; inputW < widthEnd; inputW++ {
			value := input[inputH*width+inputW]
			sum += value
			count++

			if value > maxValue {
				maxValue = value
			}
		}
	}

	if maxPool {
		return maxValue
	}

	return sum / float64(count)
}

func referenceMax(values []float64) float64 {
	maxValue := math.Inf(-1)

	for _, value := range values {
		maxValue = math.Max(maxValue, value)
	}

	return maxValue
}

func referenceSum(values []float64) float64 {
	sum := 0.0

	for _, value := range values {
		sum += value
	}

	return sum
}
