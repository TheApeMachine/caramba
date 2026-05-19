package neon

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestUnaryAbsAndExp(t *testing.T) {
	convey.Convey("abs and exp dispatch through the registry", t, func() {
		shape, _ := tensor.NewShape([]int{4})
		input, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		inputView, _ := input.Float32Native()
		copy(inputView, []float32{-1, 0, 1, 2})

		absKernel, ok := Default.Lookup("abs", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		})
		convey.So(ok, convey.ShouldBeTrue)

		err := absKernel.Run(input, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		convey.So(outView, convey.ShouldResemble, []float32{1, 0, 1, 2})

		expKernel, ok := Default.Lookup("exp", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		})
		convey.So(ok, convey.ShouldBeTrue)

		inputView[0] = 0
		err = expKernel.Run(input, out)
		convey.So(err, convey.ShouldBeNil)

		convey.So(math.Abs(float64(outView[0]-1)), convey.ShouldBeLessThan, 1e-6)
	})
}

func TestReductionSum(t *testing.T) {
	convey.Convey("sum kernel produces total", t, func() {
		shape, _ := tensor.NewShape([]int{4})
		input, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		inputView, _ := input.Float32Native()
		copy(inputView, []float32{1, 2, 3, 4})

		kernel, _ := Default.Lookup("sum", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		})

		err := kernel.Run(input, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		convey.So(outView[0], convey.ShouldEqual, float32(10))
	})
}

func TestMSELoss(t *testing.T) {
	convey.Convey("mse_loss returns mean squared error", t, func() {
		shape, _ := tensor.NewShape([]int{3})
		predictions, _ := tensor.NewZeroed(shape, dtype.Float32)
		targets, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		predView, _ := predictions.Float32Native()
		copy(predView, []float32{1, 2, 3})

		// (1-1)^2 + (2-3)^2 + (3-5)^2 = 0 + 1 + 4 = 5; mean = 5/3.
		targetView, _ := targets.Float32Native()
		copy(targetView, []float32{1, 3, 5})

		err := runMSELoss(predictions, targets, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		convey.So(math.Abs(float64(outView[0])-5.0/3.0), convey.ShouldBeLessThan, 1e-5)
	})
}

func TestGreedySample(t *testing.T) {
	convey.Convey("greedy sample picks the argmax", t, func() {
		shape, _ := tensor.NewShape([]int{5})
		logits, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(tensor.Shape{}, dtype.Int32)
		_ = out
		outShape, _ := tensor.NewShape([]int{1})
		outTensor, _ := tensor.NewZeroed(outShape, dtype.Int32)

		logitView, _ := logits.Float32Native()
		copy(logitView, []float32{0.1, 0.2, 0.9, 0.3, 0.4})

		err := runGreedySample(logits, outTensor)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := outTensor.Int32Native()
		convey.So(outView[0], convey.ShouldEqual, int32(2))
	})
}

func TestEmbeddingLookup(t *testing.T) {
	convey.Convey("embedding_lookup gathers rows", t, func() {
		tableShape, _ := tensor.NewShape([]int{3, 2})
		table, _ := tensor.NewZeroed(tableShape, dtype.Float32)
		tableView, _ := table.Float32Native()
		copy(tableView, []float32{1, 2, 3, 4, 5, 6})

		indicesShape, _ := tensor.NewShape([]int{2})
		indices, _ := tensor.NewZeroed(indicesShape, dtype.Int32)
		indicesView, _ := indices.Int32Native()
		indicesView[0] = 2
		indicesView[1] = 0

		outShape, _ := tensor.NewShape([]int{2, 2})
		out, _ := tensor.NewZeroed(outShape, dtype.Float32)

		err := runEmbeddingLookup(table, indices, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		convey.So(outView, convey.ShouldResemble, []float32{5, 6, 1, 2})
	})
}

func TestMaxPool2D(t *testing.T) {
	convey.Convey("max_pool2d picks the maximum per 2x2 window", t, func() {
		inputShape, _ := tensor.NewShape([]int{1, 1, 2, 2})
		input, _ := tensor.NewZeroed(inputShape, dtype.Float32)
		outShape, _ := tensor.NewShape([]int{1, 1, 1, 1})
		out, _ := tensor.NewZeroed(outShape, dtype.Float32)

		inputView, _ := input.Float32Native()
		copy(inputView, []float32{1, 2, 3, 4})

		err := MaxPool2DFloat32(PoolConfig{KernelH: 2, KernelW: 2, StrideH: 2, StrideW: 2}, input, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		convey.So(outView[0], convey.ShouldEqual, float32(4))
	})
}

func TestMultiHeadAttentionShape(t *testing.T) {
	convey.Convey("multi_head_attention runs without error on minimal shape", t, func() {
		shape, _ := tensor.NewShape([]int{2, 8 * 64})
		query, _ := tensor.NewZeroed(shape, dtype.Float32)
		key, _ := tensor.NewZeroed(shape, dtype.Float32)
		value, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		err := MultiHeadAttentionFloat32(
			DefaultMultiHeadAttentionConfig(),
			query, key, value, out,
		)

		convey.So(err, convey.ShouldBeNil)
	})
}
