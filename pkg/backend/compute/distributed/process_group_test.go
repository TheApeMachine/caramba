package distributed

import (
	"context"
	"sync"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestLocalProcessGroup_AllReduceSum(t *testing.T) {
	convey.Convey("Given four ranks each contributing 1, 2, 3, 4", t, func() {
		ctx := context.Background()
		groups := NewLocalProcessGroup(4)
		defer func() {
			for _, group := range groups {
				group.Close()
			}
		}()

		shape, _ := tensor.NewShape([]int{2})
		shards := make([]tensor.Tensor, 4)

		for index := range groups {
			shard, _ := tensor.NewZeroed(shape, dtype.Float32)
			view, _ := shard.Float32Native()

			view[0] = float32(index + 1)
			view[1] = float32(index + 1)

			shards[index] = shard
		}

		defer func() {
			for _, shard := range shards {
				shard.Close()
			}
		}()

		var waitGroup sync.WaitGroup

		for index, group := range groups {
			waitGroup.Add(1)

			go func(rank int, group *LocalProcessGroup) {
				defer waitGroup.Done()
				_ = group.AllReduce(ctx, OpSum, shards[rank])
			}(index, group)
		}

		waitGroup.Wait()

		convey.Convey("Every shard should hold 1+2+3+4 = 10", func() {
			for _, shard := range shards {
				view, _ := shard.Float32Native()
				convey.So(view, convey.ShouldResemble, []float32{10, 10})
			}
		})
	})
}

func TestLocalProcessGroup_Barrier(t *testing.T) {
	convey.Convey("Given three ranks calling Barrier with the same tag", t, func() {
		ctx := context.Background()
		groups := NewLocalProcessGroup(3)
		defer func() {
			for _, group := range groups {
				group.Close()
			}
		}()

		var waitGroup sync.WaitGroup
		results := make([]error, 3)

		for index, group := range groups {
			waitGroup.Add(1)

			go func(rank int, group *LocalProcessGroup) {
				defer waitGroup.Done()
				results[rank] = group.Barrier(ctx, 1)
			}(index, group)
		}

		waitGroup.Wait()

		convey.Convey("All three calls should return without error", func() {
			for _, err := range results {
				convey.So(err, convey.ShouldBeNil)
			}
		})
	})
}
