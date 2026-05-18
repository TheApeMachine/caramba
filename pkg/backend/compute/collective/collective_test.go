package collective

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestAllReduce_Sum(t *testing.T) {
	convey.Convey("Given four shards of [3] float32", t, func() {
		ctx := context.Background()
		shape, _ := tensor.NewShape([]int{3})

		shards := make([]tensor.Tensor, 4)

		for index := range shards {
			shard, _ := tensor.NewZeroed(shape, dtype.Float32)
			view, _ := shard.Float32Native()

			for elementIndex := range view {
				view[elementIndex] = float32(index + 1)
			}

			shards[index] = shard
		}

		defer func() {
			for _, shard := range shards {
				shard.Close()
			}
		}()

		convey.Convey("AllReduce(Sum) should write 1+2+3+4=10 to every shard", func() {
			err := AllReduce(ctx, OpSum, shards)
			convey.So(err, convey.ShouldBeNil)

			for _, shard := range shards {
				view, _ := shard.Float32Native()
				convey.So(view, convey.ShouldResemble, []float32{10, 10, 10})
			}
		})
	})
}

func TestAllReduce_Mean(t *testing.T) {
	convey.Convey("Given two shards [4 8]", t, func() {
		ctx := context.Background()
		shape, _ := tensor.NewShape([]int{1})

		left, _ := tensor.NewZeroed(shape, dtype.Float32)
		right, _ := tensor.NewZeroed(shape, dtype.Float32)

		leftView, _ := left.Float32Native()
		rightView, _ := right.Float32Native()

		leftView[0] = 4
		rightView[0] = 8

		err := AllReduce(ctx, OpMean, []tensor.Tensor{left, right})

		convey.Convey("Each shard should hold the mean", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(leftView[0], convey.ShouldEqual, float32(6))
			convey.So(rightView[0], convey.ShouldEqual, float32(6))
		})
	})
}

func TestBroadcast(t *testing.T) {
	convey.Convey("Given three shards", t, func() {
		ctx := context.Background()
		shape, _ := tensor.NewShape([]int{2})

		shards := make([]tensor.Tensor, 3)

		for index := range shards {
			shard, _ := tensor.NewZeroed(shape, dtype.Float32)
			view, _ := shard.Float32Native()

			view[0] = float32(index * 10)
			view[1] = float32(index*10 + 1)

			shards[index] = shard
		}

		err := Broadcast(ctx, 1, shards)

		convey.Convey("Every shard should match the source", func() {
			convey.So(err, convey.ShouldBeNil)

			for _, shard := range shards {
				view, _ := shard.Float32Native()
				convey.So(view, convey.ShouldResemble, []float32{10, 11})
			}
		})
	})
}
