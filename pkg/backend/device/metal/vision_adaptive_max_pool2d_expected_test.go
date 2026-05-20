package metal

import (
	"testing"

	cpupool "github.com/theapemachine/caramba/pkg/backend/device/cpu/pool"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

const (
	adaptiveMaxPool2DBatch      = 2
	adaptiveMaxPool2DChannels   = 2
	adaptiveMaxPool2DInputHeight  = 5
	adaptiveMaxPool2DOutputHeight = 3
)

type adaptiveMaxPool2DFixture struct {
	inputBytes    []byte
	expectedBytes []byte
}

func adaptiveMaxPool2DOutputWidthForTest(widthCase int) int {
	if widthCase < 1 {
		return 1
	}

	return widthCase
}

func adaptiveMaxPool2DInputWidthForTest(outputWidth int) int {
	return outputWidth + 3
}

func adaptiveMaxPool2DFixtureForTest(outputWidth int, storageDType dtype.DType) adaptiveMaxPool2DFixture {
	inputHeight := adaptiveMaxPool2DInputHeight
	inputWidth := adaptiveMaxPool2DInputWidthForTest(outputWidth)
	outHeight := adaptiveMaxPool2DOutputHeight
	outWidth := outputWidth

	inputShape, err := tensor.NewShape([]int{
		adaptiveMaxPool2DBatch, adaptiveMaxPool2DChannels, inputHeight, inputWidth,
	})
	if err != nil {
		panic(err)
	}

	outShape, err := tensor.NewShape([]int{
		adaptiveMaxPool2DBatch, adaptiveMaxPool2DChannels, outHeight, outWidth,
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
		adaptiveMaxPool2DBatch*adaptiveMaxPool2DChannels*inputHeight*inputWidth, 61, 64,
	)
	inputView, err := hostInput.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(inputView, inputValues)

	runAdaptiveMaxPool2DScalarReference(hostInput, hostOutput)

	outputView, err := hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	if storageDType == dtype.Float32 {
		return adaptiveMaxPool2DFixture{
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

	runAdaptiveMaxPool2DScalarReference(roundTripInput, hostOutput)

	outputView, err = hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	return adaptiveMaxPool2DFixture{
		inputBytes:    inputBytes,
		expectedBytes: encodeFloat32ValuesAsDType(outputView, storageDType),
	}
}

func runAdaptiveMaxPool2DScalarReference(hostInput tensor.Tensor, hostOutput tensor.Tensor) {
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
		true,
	)
}

func adaptiveMaxPool2DDTypeBytes(outputWidth int, storageDType dtype.DType) ([]byte, []byte) {
	fixture := adaptiveMaxPool2DFixtureForTest(outputWidth, storageDType)

	return fixture.inputBytes, fixture.expectedBytes
}

func adaptiveMaxPool2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputHeight := adaptiveMaxPool2DInputHeight
	inputWidth := adaptiveMaxPool2DInputWidthForTest(outputWidth)
	outHeight := adaptiveMaxPool2DOutputHeight
	outWidth := outputWidth
	inputShape := mustShapeForTest(testingObject, []int{
		adaptiveMaxPool2DBatch, adaptiveMaxPool2DChannels, inputHeight, inputWidth,
	})
	outShape := mustShapeForTest(testingObject, []int{
		adaptiveMaxPool2DBatch, adaptiveMaxPool2DChannels, outHeight, outWidth,
	})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, out
}
