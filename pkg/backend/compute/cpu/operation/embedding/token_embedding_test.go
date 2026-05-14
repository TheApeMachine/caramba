package embedding

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestTokenEmbedding(t *testing.T) {
	Convey("Given a TokenEmbedding operation", t, func() {
		op := NewTokenEmbedding(4, 8, 0.02)
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

				outputState, err := op.Forward(stateDict)

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

				outputState, err := op.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out[:3], ShouldResemble, outputState.Out[3:])
			})
		})
	})
}

func BenchmarkTokenEmbedding_Forward(b *testing.B) {
	op := NewTokenEmbedding(32000, 512, 0.02)
	tokens := make([]float64, 512)
	weight := make([]float64, 32000*512)

	for index := range tokens {
		tokens[index] = float64(index % 32000)
	}

	for index := range weight {
		weight[index] = float64(index%37) * 0.001
	}

	for b.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{1, 512}).
			WithInput(tokens).
			WithWeight(weight)
		stateDict.VocabSize = 32000
		stateDict.DModel = 512
		_, _ = op.Forward(stateDict)
	}
}
