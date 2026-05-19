package neon

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/attention"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/elementwise"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/embedding"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/losses"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/pool"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/reduction"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/sampling"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestUnaryAbsAndExp(t *testing.T) {
	convey.Convey("abs and exp dispatch through domain ops", t, func() {
		shape, _ := tensor.NewShape([]int{4})
		input, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		inputView, _ := input.Float32Native()
		copy(inputView, []float32{-1, 0, 1, 2})

		outView, _ := out.Float32Native()
		elementwise.Abs(
			unsafe.Pointer(&outView[0]),
			unsafe.Pointer(&inputView[0]),
			len(inputView),
			dtype.Float32,
		)

		convey.So(outView, convey.ShouldResemble, []float32{1, 0, 1, 2})

		inputView[0] = 0
		activation.Exp(
			unsafe.Pointer(&outView[0]),
			unsafe.Pointer(&inputView[0]),
			len(inputView),
			dtype.Float32,
		)

		convey.So(parity.Float32ULPDistance(outView[0], 1), convey.ShouldBeLessThanOrEqualTo, 2)
	})
}

func TestReductionSum(t *testing.T) {
	convey.Convey("sum produces total", t, func() {
		shape, _ := tensor.NewShape([]int{4})
		input, _ := tensor.NewZeroed(shape, dtype.Float32)

		inputView, _ := input.Float32Native()
		copy(inputView, []float32{1, 2, 3, 4})

		total := reduction.Sum(
			unsafe.Pointer(&inputView[0]),
			len(inputView),
			dtype.Float32,
		)

		convey.So(total, convey.ShouldEqual, float32(10))
	})
}

func TestMSELoss(t *testing.T) {
	convey.Convey("mse returns mean squared error", t, func() {
		shape, _ := tensor.NewShape([]int{3})
		predictions, _ := tensor.NewZeroed(shape, dtype.Float32)
		targets, _ := tensor.NewZeroed(shape, dtype.Float32)

		predView, _ := predictions.Float32Native()
		copy(predView, []float32{1, 2, 3})

		targetView, _ := targets.Float32Native()
		copy(targetView, []float32{1, 3, 5})

		mean := losses.MSE(
			unsafe.Pointer(&predView[0]),
			unsafe.Pointer(&targetView[0]),
			len(predView),
			dtype.Float32,
		)

		convey.So(parity.Float32ULPDistance(mean, float32(5.0/3.0)), convey.ShouldBeLessThanOrEqualTo, 4)
	})
}

func TestGreedySample(t *testing.T) {
	convey.Convey("greedy sample picks the argmax", t, func() {
		shape, _ := tensor.NewShape([]int{5})
		logits, _ := tensor.NewZeroed(shape, dtype.Float32)

		logitView, _ := logits.Float32Native()
		copy(logitView, []float32{0.1, 0.2, 0.9, 0.3, 0.4})

		token := sampling.GreedySample(unsafe.Pointer(&logitView[0]), len(logitView), dtype.Float32)
		convey.So(token, convey.ShouldEqual, int32(2))
	})
}

func TestEmbeddingLookup(t *testing.T) {
	convey.Convey("embedding lookup gathers rows", t, func() {
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
		outView, _ := out.Float32Native()

		embedding.Lookup(
			unsafe.Pointer(&tableView[0]),
			unsafe.Pointer(&indicesView[0]),
			unsafe.Pointer(&outView[0]),
			3, 2, 2,
			dtype.Float32,
		)

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

		err := pool.MaxPool2DFloat32(pool.PoolConfig{KernelH: 2, KernelW: 2, StrideH: 2, StrideW: 2}, input, out)
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

		err := attention.MultiHeadAttentionFloat32(
			attention.DefaultMultiHeadAttentionConfig(),
			query, key, value, out,
		)

		convey.So(err, convey.ShouldBeNil)
	})
}
