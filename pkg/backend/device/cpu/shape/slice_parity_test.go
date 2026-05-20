package shape

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunSlice(t *testing.T) {
	for _, length := range parity.Lengths {
		length := length

		t.Run(fmt.Sprintf("N=%d", length), func(t *testing.T) {
			convey.Convey("Given scalar slice along dim 1 with start 0", t, func() {
				runSliceParityCase(t, length)
			})
		})
	}
}

func runSliceParityCase(testingObject *testing.T, seqLen int) {
	const inner = 4

	const tail = 7

	inputShape, err := tensor.NewShape([]int{1, seqLen + tail, inner})
	convey.So(err, convey.ShouldBeNil)

	outShape, err := tensor.NewShape([]int{1, seqLen, inner})
	convey.So(err, convey.ShouldBeNil)

	hostInput, err := tensor.NewZeroed(inputShape, dtype.Float32)
	convey.So(err, convey.ShouldBeNil)

	hostOutput, err := tensor.NewZeroed(outShape, dtype.Float32)
	convey.So(err, convey.ShouldBeNil)

	wantOutput, err := tensor.NewZeroed(outShape, dtype.Float32)
	convey.So(err, convey.ShouldBeNil)

	inputView, err := hostInput.Float32Native()
	convey.So(err, convey.ShouldBeNil)

	for index := range inputView {
		inputView[index] = float32(index)*0.01 - 1
	}

	wantView, err := wantOutput.Float32Native()
	convey.So(err, convey.ShouldBeNil)

	sliceContiguousReference(wantView, inputView, inputShape.Dims(), 1, 0, seqLen)

	dimTensor, err := newInt32ScalarTensor(1)
	convey.So(err, convey.ShouldBeNil)

	startTensor, err := newInt32ScalarTensor(0)
	convey.So(err, convey.ShouldBeNil)

	endTensor, err := newInt32ScalarTensor(int32(seqLen))
	convey.So(err, convey.ShouldBeNil)

	err = RunSlice(hostInput, dimTensor, startTensor, endTensor, hostOutput)
	convey.So(err, convey.ShouldBeNil)

	gotView, err := hostOutput.Float32Native()
	convey.So(err, convey.ShouldBeNil)

	parity.AssertFloat32SlicesWithinULP(testingObject, gotView, wantView, 0)
}

func sliceContiguousReference(
	out []float32,
	in []float32,
	inDims []int,
	dim int,
	start int,
	end int,
) {
	outer := 1
	for axis := 0; axis < dim; axis++ {
		outer *= inDims[axis]
	}

	inner := 1
	for axis := dim + 1; axis < len(inDims); axis++ {
		inner *= inDims[axis]
	}

	inputStride := inDims[dim] * inner
	blockLen := (end - start) * inner

	for outerIndex := 0; outerIndex < outer; outerIndex++ {
		inOffset := outerIndex*inputStride + start*inner
		outOffset := outerIndex * blockLen

		copy(out[outOffset:outOffset+blockLen], in[inOffset:inOffset+blockLen])
	}
}

func newInt32ScalarTensor(value int32) (tensor.Tensor, error) {
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
