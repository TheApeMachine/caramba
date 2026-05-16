package attention

import (
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"

	. "github.com/smartystreets/goconvey/convey"
)

func TestGQA(test *testing.T) {
	Convey("Given grouped-query attention", test, func() {
		operation := NewGQA()

		Convey("It should accept rank-four query shape with configured KV heads", func() {
			query := []float64{
				1, 0,
				0, 1,
				1, 1,
				1, -1,
			}
			key := []float64{
				1, 0,
				0, 1,
			}
			value := []float64{
				2, 3,
				5, 7,
			}
			dict := state.NewDict().WithShape([]int{1, 2, 2, 2}).WithInputs(query, key, value)
			dict.NumKVHeads = 1
			dict.HeadDim = 2

			outputState, err := operation.Forward(dict)

			So(err, ShouldBeNil)
			So(outputState.Out, ShouldHaveLength, len(query))
		})

		Convey("It should keep legacy rank-five shape support", func() {
			query := make([]float64, 1*2*2*2)
			key := make([]float64, 1*1*2*2)
			value := make([]float64, 1*1*2*2)
			dict := state.NewDict().WithShape([]int{1, 2, 1, 2, 2}).WithInputs(
				query,
				key,
				value,
			)

			outputState, err := operation.Forward(dict)

			So(err, ShouldBeNil)
			So(outputState.Out, ShouldHaveLength, len(query))
		})

		Convey("It should attend over cached KV history during incremental decode", func() {
			cache := kv.NewCache()
			first := state.NewDict().WithShape([]int{1, 2, 1, 2}).WithInputs(
				[]float64{1, 0, 0, 1},
				[]float64{1, 0},
				[]float64{2, 3},
			)
			first.NodeID = "attn_0"
			first.NumKVHeads = 1
			first.HeadDim = 2
			first.Causal = true
			first.KVCache = cache

			_, err := operation.Forward(first)
			So(err, ShouldBeNil)

			query := []float64{0, 1, 1, 0}
			key := []float64{0, 1}
			value := []float64{5, 7}
			second := state.NewDict().WithShape([]int{1, 2, 1, 2}).WithInputs(
				query,
				key,
				value,
			)
			second.NodeID = "attn_0"
			second.NumKVHeads = 1
			second.HeadDim = 2
			second.Causal = true
			second.KVCache = cache

			outputState, err := operation.Forward(second)
			So(err, ShouldBeNil)

			expected := make([]float64, len(query))
			cachedKey := []float64{1, 0, 0, 1}
			cachedValue := []float64{2, 3, 5, 7}
			sdpaHeadCausal(expected[0:2], query[0:2], cachedKey, cachedValue, 1, 2, 2)
			sdpaHeadCausal(expected[2:4], query[2:4], cachedKey, cachedValue, 1, 2, 2)
			So(outputState.Out, ShouldResemble, expected)
		})
	})
}

func BenchmarkGQA_Forward(benchmark *testing.B) {
	operation := NewGQA()
	query := make([]float64, 1*32*128*64)
	key := make([]float64, 1*8*128*64)
	value := make([]float64, 1*8*128*64)

	for index := range query {
		query[index] = float64(index%17) / 17
	}

	for index := range key {
		key[index] = float64(index%13) / 13
		value[index] = float64(index%11) / 11
	}

	for benchmark.Loop() {
		dict := state.NewDict().WithShape([]int{1, 32, 128, 64}).WithInputs(
			query,
			key,
			value,
		)
		dict.NumKVHeads = 8
		dict.HeadDim = 64

		_, _ = operation.Forward(dict)
	}
}
