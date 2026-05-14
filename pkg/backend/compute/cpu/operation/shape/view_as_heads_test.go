package shape

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestViewAsHeads(t *testing.T) {
	Convey("Given a ViewAsHeads operation", t, func() {
		op := NewViewAsHeads()

		Convey("Forward", func() {
			Convey("It should lay out [B,T,D] as [B,H,T,head_dim]", func() {
				input := []float64{
					1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12,
					13, 14, 15, 16,
				}
				stateDict := state.NewDict().
					WithShape([]int{2, 2, 4}).
					WithInput(input)
				stateDict.NumHeads = 2

				outputState, err := op.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{
					1, 2, 5, 6,
					3, 4, 7, 8,
					9, 10, 13, 14,
					11, 12, 15, 16,
				})
			})
		})
	})
}

func BenchmarkViewAsHeads_Forward(b *testing.B) {
	op := NewViewAsHeads()
	shape := []int{4, 512, 1024}
	input := make([]float64, 4*512*1024)

	for index := range input {
		input[index] = float64(index)
	}

	for b.Loop() {
		stateDict := state.NewDict().
			WithShape(shape).
			WithInput(input)
		stateDict.NumHeads = 16
		_, _ = op.Forward(stateDict)
	}
}
