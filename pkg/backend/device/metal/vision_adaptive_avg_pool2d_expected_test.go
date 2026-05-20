package metal

import (
	"testing"

	cpupool "github.com/theapemachine/caramba/pkg/backend/device/cpu/pool"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

const (
	adaptiveAvgPool2DBatch        = 2
	adaptiveAvgPool2DChannels     = 2
	adaptiveAvgPool2DInputHeight  = 5
	adaptiveAvgPool2DOutputHeight = 3
)

type adaptiveAvgPool2DFixture struct {
	inputBytes    []byte
	expectedBytes []byte
}

func adaptiveAvgPool2DOutputWidthForTest(widthCase int) int {
	if widthCase < 1 {
		return 1
	}

	return widthCase
}

func adaptiveAvgPool2DInputWidthForTest(outputWidth int) int {
	return outputWidth + 3
}

func adaptiveAvgPool2DFixtureForTest(outputWidth int, storageDType dtype.DType) adaptiveAvgPool2DFixture {
	inputHeight := adaptiveAvgPool2DInputHeight
	inputWidth := adaptiveAvgPool2DInputWidthForTest(outputWidth)
	outHeight := adaptiveAvgPool2DOutputHeight
	outWidth := outputWidth

	inputShape, err := tensor.NewShape([]int{
		adaptiveAvgPool2DBatch, adaptiveAvgPool2DChannels, inputHeight, inputWidth,
	})
	if err != nil {
		panic(err)
	}

	outShape, err := tensor.NewShape([]int{
		adaptiveAvgPool2DBatch, adaptiveAvgPool2DChannels, outHeight, outWidth,
	})
	if err != nil {
		panic(err)
	}

	hostInput, err := tensor.NewZeroed(inputShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	hostOutput, err := tensor.NewZeroed(outShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	inputValues := projectionValues(
		adaptiveAvgPool2DBatch*adaptiveAvgPool2DChannels*inputHeight*inputWidth, 61, 64,
	)
	inputView, err := hostInput.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(inputView, inputValues)

	runAdaptiveAvgPool2DScalarReference(hostInput, hostOutput)

	outputView, err := hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	if storageDType == dtype.Float32 {
		return adaptiveAvgPool2DFixture{
			inputBytes:    dtypeconvert.Float32ToBytes(inputView),
			expectedBytes: dtypeconvert.Float32ToBytes(outputView),
		}
	}

	inputBytes := encodeFloat32ValuesAsDType(inputView, storageDType)
	storedInput := decodeDTypeBytesToFloat32(inputBytes, storageDType)

	roundTripInput, err := tensor.NewZeroed(inputShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	roundTripInputView, err := roundTripInput.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(roundTripInputView, storedInput)

	runAdaptiveAvgPool2DScalarReference(roundTripInput, hostOutput)

	outputView, err = hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	return adaptiveAvgPool2DFixture{
		inputBytes:    inputBytes,
		expectedBytes: encodeFloat32ValuesAsDType(outputView, storageDType),
	}
}

func runAdaptiveAvgPool2DScalarReference(hostInput tensor.Tensor, hostOutput tensor.Tensor) {
	inputView, err := hostInput.Float32Native()
	if err != nil {
		panic(err)
	}

	outputView, err := hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	inDims := hostInput.Shape().Dims()
	outDims := hostOutput.Shape().Dims()

	cpupool.AdaptivePool2DFloat32Scalar(
		inputView,
		outputView,
		inDims[0], inDims[1], inDims[2], inDims[3],
		outDims[2], outDims[3],
		false,
	)
}

func adaptiveAvgPool2DDTypeBytes(outputWidth int, storageDType dtype.DType) ([]byte, []byte) {
	fixture := adaptiveAvgPool2DFixtureForTest(outputWidth, storageDType)

	return fixture.inputBytes, fixture.expectedBytes
}

func adaptiveAvgPool2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputHeight := adaptiveAvgPool2DInputHeight
	inputWidth := adaptiveAvgPool2DInputWidthForTest(outputWidth)
	outHeight := adaptiveAvgPool2DOutputHeight
	outWidth := outputWidth
	inputShape := mustShapeForTest(testingObject, []int{
		adaptiveAvgPool2DBatch, adaptiveAvgPool2DChannels, inputHeight, inputWidth,
	})
	outShape := mustShapeForTest(testingObject, []int{
		adaptiveAvgPool2DBatch, adaptiveAvgPool2DChannels, outHeight, outWidth,
	})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, out
}
