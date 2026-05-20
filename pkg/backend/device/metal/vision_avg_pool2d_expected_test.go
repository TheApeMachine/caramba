package metal

import (
	"testing"

	cpupool "github.com/theapemachine/caramba/pkg/backend/device/cpu/pool"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

const avgPool2DInputHeight = 4

type avgPool2DFixture struct {
	inputBytes    []byte
	expectedBytes []byte
}

func avgPool2DOutputWidthForTest(widthCase int) int {
	if widthCase < 1 {
		return 1
	}

	return widthCase
}

func avgPool2DInputWidthForTest(outputWidth int) int {
	return outputWidth * 2
}

func avgPool2DFixtureForTest(outputWidth int, storageDType dtype.DType) avgPool2DFixture {
	batch, channels := 2, 3
	inputHeight := avgPool2DInputHeight
	inputWidth := avgPool2DInputWidthForTest(outputWidth)
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

	runAvgPool2DScalarReference(hostInput, hostOutput)

	outputView, err := hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	if storageDType == dtype.Float32 {
		return avgPool2DFixture{
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

	runAvgPool2DScalarReference(roundTripInput, hostOutput)

	outputView, err = hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	return avgPool2DFixture{
		inputBytes:    inputBytes,
		expectedBytes: encodeFloat32ValuesAsDType(outputView, storageDType),
	}
}

func runAvgPool2DScalarReference(hostInput tensor.Tensor, hostOutput tensor.Tensor) {
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
		false,
	)
}

func avgPool2DDTypeBytes(outputWidth int, storageDType dtype.DType) ([]byte, []byte) {
	fixture := avgPool2DFixtureForTest(outputWidth, storageDType)

	return fixture.inputBytes, fixture.expectedBytes
}

func avgPool2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, channels := 2, 3
	inputHeight := avgPool2DInputHeight
	inputWidth := avgPool2DInputWidthForTest(outputWidth)
	outHeight := inputHeight / 2
	outWidth := outputWidth
	inputShape := mustShapeForTest(testingObject, []int{batch, channels, inputHeight, inputWidth})
	outShape := mustShapeForTest(testingObject, []int{batch, channels, outHeight, outWidth})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, out
}
