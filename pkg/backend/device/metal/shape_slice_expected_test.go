package metal

import (
	cpushape "github.com/theapemachine/caramba/pkg/backend/device/cpu/shape"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

const sliceInnerDim = 4

const sliceTailDim = 7

type sliceFixture struct {
	inputBytes    []byte
	expectedBytes []byte
}

func sliceFixtureForTest(seqLen int, storageDType dtype.DType) sliceFixture {
	inputShape, err := tensor.NewShape([]int{1, seqLen + sliceTailDim, sliceInnerDim})
	if err != nil {
		panic(err)
	}

	outShape, err := tensor.NewShape([]int{1, seqLen, sliceInnerDim})
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

	inputView, err := hostInput.Float32Native()
	if err != nil {
		panic(err)
	}

	for index := range inputView {
		inputView[index] = float32(index)*0.01 - 1
	}

	dimTensor, err := hostInt32ScalarTensor(1)
	if err != nil {
		panic(err)
	}

	startTensor, err := hostInt32ScalarTensor(0)
	if err != nil {
		panic(err)
	}

	endTensor, err := hostInt32ScalarTensor(int32(seqLen))
	if err != nil {
		panic(err)
	}

	if err := cpushape.RunSlice(hostInput, dimTensor, startTensor, endTensor, hostOutput); err != nil {
		panic(err)
	}

	outputView, err := hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	if storageDType == dtype.Float32 {
		return sliceFixture{
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

	roundTripView, err := roundTripInput.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(roundTripView, storedInput)

	if err := cpushape.RunSlice(roundTripInput, dimTensor, startTensor, endTensor, hostOutput); err != nil {
		panic(err)
	}

	outputView, err = hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	return sliceFixture{
		inputBytes:    inputBytes,
		expectedBytes: encodeFloat32ValuesAsDType(outputView, storageDType),
	}
}

func hostInt32ScalarTensor(value int32) (tensor.Tensor, error) {
	shape, err := tensor.NewShape([]int{1})
	if err != nil {
		return nil, err
	}

	hostTensor, err := tensor.NewZeroed(shape, dtype.Int32)
	if err != nil {
		return nil, err
	}

	view, err := hostTensor.Int32Native()
	if err != nil {
		return nil, err
	}

	view[0] = value

	return hostTensor, nil
}

func sliceInputShapeForTest(seqLen int) []int {
	return []int{1, seqLen + sliceTailDim, sliceInnerDim}
}

func sliceOutputShapeForTest(seqLen int) []int {
	return []int{1, seqLen, sliceInnerDim}
}
