package embedding

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestTokenEmbedding(test *testing.T) {
	Convey("Given a TokenEmbedding operation", test, func() {
		operation := NewTokenEmbedding(4, 8, 0.02)
		weight := []float64{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		}

		Convey("Forward", func() {
			Convey("It should return the correct embedding vectors from the state dict", func() {
				stateDict := state.NewDict().
					WithShape([]int{1, 4}).
					WithInput([]float64{0, 1, 2, 3}).
					WithWeight(weight)
				stateDict.VocabSize = 4
				stateDict.DModel = 3

				outputState, err := operation.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, weight)
			})

			Convey("It should return the same vector for the same token id", func() {
				stateDict := state.NewDict().
					WithShape([]int{1, 2}).
					WithInput([]float64{2, 2}).
					WithWeight(weight)
				stateDict.VocabSize = 4
				stateDict.DModel = 3

				outputState, err := operation.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out[:3], ShouldResemble, outputState.Out[3:])
			})

			Convey("It should match reference lookup across SIMD copy lengths", func() {
				for _, dModel := range []int{1, 7, 64, 1024, 8192} {
					vocabSize := 17
					tokens := []float64{16, 0, 9, 3, 3, 12, 1}
					weight := make([]float64, vocabSize*dModel)

					for index := range weight {
						weight[index] = float64((index*37)%1009) / 1009.0
					}

					stateDict := state.NewDict().
						WithShape([]int{1, len(tokens)}).
						WithInput(tokens).
						WithWeight(weight)
					stateDict.VocabSize = vocabSize
					stateDict.DModel = dModel

					outputState, err := operation.Forward(stateDict)

					So(err, ShouldBeNil)
					So(outputState.Out, ShouldResemble, tokenEmbeddingReference(tokens, weight, dModel))
				}
			})
		})
	})
}

func BenchmarkTokenEmbedding_Forward(benchmark *testing.B) {
	operation := NewTokenEmbedding(32000, 512, 0.02)
	tokens := make([]float64, 512)
	weight := make([]float64, 32000*512)

	for index := range tokens {
		tokens[index] = float64(index % 32000)
	}

	for index := range weight {
		weight[index] = float64(index%37) * 0.001
	}

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{1, 512}).
			WithInput(tokens).
			WithWeight(weight)
		stateDict.VocabSize = 32000
		stateDict.DModel = 512
		_, _ = operation.Forward(stateDict)
	}
}

func BenchmarkTokenEmbedding_ForwardCopyLengths(benchmark *testing.B) {
	for _, dModel := range []int{1, 7, 64, 1024, 8192} {
		benchmark.Run(fmt.Sprintf("d_model_%d", dModel), func(benchmark *testing.B) {
			operation := NewTokenEmbedding(4096, dModel, 0.02)
			tokens := make([]float64, 128)
			weight := make([]float64, 4096*dModel)

			for index := range tokens {
				tokens[index] = float64((index * 17) % 4096)
			}

			for index := range weight {
				weight[index] = float64(index%97) * 0.001
			}

			stateDict := state.NewDict().
				WithShape([]int{1, len(tokens)}).
				WithInput(tokens).
				WithWeight(weight)
			stateDict.VocabSize = 4096
			stateDict.DModel = dModel

			benchmark.ResetTimer()

			for benchmark.Loop() {
				_, _ = operation.Forward(stateDict)
			}
		})
	}
}

func tokenEmbeddingReference(tokens, weight []float64, dModel int) []float64 {
	output := make([]float64, len(tokens)*dModel)

	for tokenIndex, token := range tokens {
		tokenID := int(token)
		copy(
			output[tokenIndex*dModel:(tokenIndex+1)*dModel],
			weight[tokenID*dModel:(tokenID+1)*dModel],
		)
	}

	return output
}
