package shape

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestMergeHeads(t *testing.T) {
	Convey("Given a MergeHeads operation", t, func() {
		op := NewMergeHeads()

		Convey("Forward", func() {
			Convey("It should lay out [B,H,T,head_dim] as [B,T,H,head_dim]", func() {
				input := []float64{
					1, 2, 5, 6,
					3, 4, 7, 8,
					9, 10, 13, 14,
					11, 12, 15, 16,
				}

				outputState, err := op.Forward(
					state.NewDict().
						WithShape([]int{2, 2, 2, 2}).
						WithInput(input),
				)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{
					1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12,
					13, 14, 15, 16,
				})
			})
		})
	})
}

func BenchmarkMergeHeads_Forward(b *testing.B) {
	op := NewMergeHeads()
	shape := []int{4, 16, 512, 64}
	input := make([]float64, 4*16*512*64)

	for index := range input {
		input[index] = float64(index)
	}

	for b.Loop() {
		_, _ = op.Forward(
			state.NewDict().
				WithShape(shape).
				WithInput(input),
		)
	}
}
