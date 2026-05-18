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
		defer left.Close()

		right, _ := tensor.NewZeroed(shape, dtype.Float32)
		defer right.Close()

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

func benchmarkAllReduceOp(b *testing.B, op Op, shardCount int) {
	ctx := context.Background()
	shape, _ := tensor.NewShape([]int{1024})

	shards := make([]tensor.Tensor, shardCount)

	for index := range shards {
		shard, _ := tensor.NewZeroed(shape, dtype.Float32)
		view, _ := shard.Float32Native()

		for valueIndex := range view {
			view[valueIndex] = float32(valueIndex+1) * 0.25
		}

		shards[index] = shard
	}

	defer func() {
		for _, shard := range shards {
			shard.Close()
		}
	}()

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(1024 * 4 * shardCount))

	for b.Loop() {
		if err := AllReduce(ctx, op, shards); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAllReduce_Sum(b *testing.B)   { benchmarkAllReduceOp(b, OpSum, 4) }
func BenchmarkAllReduce_Mean(b *testing.B)  { benchmarkAllReduceOp(b, OpMean, 4) }
func BenchmarkAllReduce_Max(b *testing.B)   { benchmarkAllReduceOp(b, OpMax, 4) }

func BenchmarkBroadcast_4(b *testing.B) {
	ctx := context.Background()
	shape, _ := tensor.NewShape([]int{1024})

	shards := make([]tensor.Tensor, 4)

	for index := range shards {
		shard, _ := tensor.NewZeroed(shape, dtype.Float32)
		shards[index] = shard
	}

	defer func() {
		for _, shard := range shards {
			shard.Close()
		}
	}()

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(1024 * 4))

	for b.Loop() {
		if err := Broadcast(ctx, 0, shards); err != nil {
			b.Fatal(err)
		}
	}
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

		defer func() {
			for _, shard := range shards {
				shard.Close()
			}
		}()

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
