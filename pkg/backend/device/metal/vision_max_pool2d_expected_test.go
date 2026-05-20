package metal

import (
	"testing"

	cpupool "github.com/theapemachine/caramba/pkg/backend/device/cpu/pool"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

const maxPool2DInputHeight = 4

type maxPool2DFixture struct {
	inputBytes    []byte
	expectedBytes []byte
}

func maxPool2DOutputWidthForTest(widthCase int) int {
	if widthCase < 1 {
		return 1
	}

	return widthCase
}

func maxPool2DInputWidthForTest(outputWidth int) int {
	return outputWidth * 2
}

func maxPool2DFixtureForTest(outputWidth int, storageDType dtype.DType) maxPool2DFixture {
	batch, channels := 2, 3
	inputHeight := maxPool2DInputHeight
	inputWidth := maxPool2DInputWidthForTest(outputWidth)
	outHeight := inputHeight / 2
	outWidth := outputWidth

	inputShape, err := tensor.NewShape([]int{batch, channels, inputHeight, inputWidth})
	if err != nil {
		panic(err)
	}

	outShape, err := tensor.NewShape([]int{batch, channels, outHeight, outWidth})
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

	inputValues := projectionValues(batch*channels*inputHeight*inputWidth, 53, 64)
	inputView, err := hostInput.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(inputView, inputValues)

	runMaxPool2DScalarReference(hostInput, hostOutput)

	outputView, err := hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	if storageDType == dtype.Float32 {
		return maxPool2DFixture{
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

	runMaxPool2DScalarReference(roundTripInput, hostOutput)

	outputView, err = hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	return maxPool2DFixture{
		inputBytes:    inputBytes,
		expectedBytes: encodeFloat32ValuesAsDType(outputView, storageDType),
	}
}

func runMaxPool2DScalarReference(hostInput tensor.Tensor, hostOutput tensor.Tensor) {
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
	config := cpupool.DefaultPoolConfig()

	cpupool.Pool2DFloat32Scalar(
		config,
		inputView,
		outputView,
		inDims[0], inDims[1], inDims[2], inDims[3],
		outDims[2], outDims[3],
		true,
	)
}

func maxPool2DDTypeBytes(outputWidth int, storageDType dtype.DType) ([]byte, []byte) {
	fixture := maxPool2DFixtureForTest(outputWidth, storageDType)

	return fixture.inputBytes, fixture.expectedBytes
}

func maxPool2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, channels := 2, 3
	inputHeight := maxPool2DInputHeight
	inputWidth := maxPool2DInputWidthForTest(outputWidth)
	outHeight := inputHeight / 2
	outWidth := outputWidth
	inputShape := mustShapeForTest(testingObject, []int{batch, channels, inputHeight, inputWidth})
	outShape := mustShapeForTest(testingObject, []int{batch, channels, outHeight, outWidth})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, out
}
