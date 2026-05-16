package attention

import (
	"math"
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

		Convey("It should match scalar reference for Llama-sized head dimensions", func() {
			batch, heads, kvHeads, seqLen, headDim := 1, 32, 8, 4, 64
			query := deterministicAttentionValues(batch*heads*seqLen*headDim, 17)
			key := deterministicAttentionValues(batch*kvHeads*seqLen*headDim, 13)
			value := deterministicAttentionValues(batch*kvHeads*seqLen*headDim, 11)
			dict := state.NewDict().
				WithShape([]int{batch, heads, seqLen, headDim}).
				WithInputs(query, key, value)
			dict.NumKVHeads = kvHeads
			dict.HeadDim = headDim
			dict.Causal = true

			outputState, err := operation.Forward(dict)

			So(err, ShouldBeNil)

			expected := scalarGQAReference(
				query,
				key,
				value,
				batch,
				heads,
				kvHeads,
				seqLen,
				headDim,
				true,
			)
			for index := range expected {
				So(math.Abs(outputState.Out[index]-expected[index]), ShouldBeLessThan, 1e-9)
			}
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

func deterministicAttentionValues(length int, period int) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = float64(index%period-period/2) / float64(period)
	}

	return values
}

func scalarGQAReference(
	query, key, value []float64,
	batch int,
	numHeads int,
	numKVHeads int,
	seqLen int,
	headDim int,
	causal bool,
) []float64 {
	output := make([]float64, len(query))
	groupSize := numHeads / numKVHeads
	scale := 1 / math.Sqrt(float64(headDim))

	for batchIndex := range batch {
		for headIndex := range numHeads {
			kvHead := headIndex / groupSize
			for queryIndex := range seqLen {
				queryOffset := ((batchIndex*numHeads+headIndex)*seqLen + queryIndex) * headDim
				keyOffset := ((batchIndex*numKVHeads + kvHead) * seqLen) * headDim
				visible := seqLen

				if causal {
					visible = queryIndex + 1
				}

				scores := make([]float64, visible)
				maxScore := math.Inf(-1)

				for keyIndex := range visible {
					score := 0.0

					for dimIndex := range headDim {
						score += query[queryOffset+dimIndex] *
							key[keyOffset+keyIndex*headDim+dimIndex]
					}

					score *= scale
					scores[keyIndex] = score

					if score > maxScore {
						maxScore = score
					}
				}

				sum := 0.0

				for scoreIndex, score := range scores {
					weight := math.Exp(score - maxScore)
					scores[scoreIndex] = weight
					sum += weight
				}

				outputOffset := queryOffset

				for keyIndex, score := range scores {
					weight := score / sum

					for dimIndex := range headDim {
						output[outputOffset+dimIndex] += weight *
							value[keyOffset+keyIndex*headDim+dimIndex]
					}
				}
			}
		}
	}

	return output
}
