package masking

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestCausalMask(t *testing.T) {
	Convey("Given a CausalMask operation", t, func() {
		op := NewCausalMask()

		Convey("Forward", func() {
			Convey("It should produce a lower-triangular mask of zeros with -inf above diagonal", func() {
				seq := 4
				out := forwardCausalMask(op, []int{seq})
				So(out, ShouldHaveLength, seq*seq)
				for row := 0; row < seq; row++ {
					for col := 0; col < seq; col++ {
						value := out[row*seq+col]
						if col <= row {
							So(value, ShouldEqual, 0.0)
						} else {
							So(math.IsInf(value, -1), ShouldBeTrue)
						}
					}
				}
			})

			Convey("It should produce a 1x1 mask of zero for seq_len=1", func() {
				out := forwardCausalMask(op, []int{1})
				So(out, ShouldResemble, []float64{0.0})
			})
		})
	})
}

func BenchmarkCausalMask_Forward(b *testing.B) {
	op := NewCausalMask()
	shape := []int{512}

	for b.Loop() {
		stateDict := state.NewDict().WithShape(shape)
		_, _ = op.Forward(stateDict)
	}
}

func forwardCausalMask(op *CausalMask, shape []int) []float64 {
	stateDict := state.NewDict().WithShape(shape)
	outputState, err := op.Forward(stateDict)

	So(err, ShouldBeNil)

	return outputState.Out
}
