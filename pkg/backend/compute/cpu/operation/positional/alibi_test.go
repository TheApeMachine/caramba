package positional

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestALiBi(t *testing.T) {
	Convey("Given an ALiBi operation", t, func() {
		op := NewALiBi()

		Convey("Forward", func() {
			Convey("It should build causal per-head linear bias rows", func() {
				stateDict := state.NewDict().WithShape([]int{2, 3, 4})
				stateDict.Causal = true

				outputState, err := op.Forward(
					stateDict,
				)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldHaveLength, 24)

				slopes := buildSlopes(2)

				for headIndex := range 2 {
					for queryIndex := range 3 {
						for keyIndex := range 4 {
							index := (headIndex*3+queryIndex)*4 + keyIndex
							expected := slopes[headIndex] * float64(keyIndex-queryIndex)
							So(math.Abs(outputState.Out[index]-expected), ShouldBeLessThan, 1e-12)
						}
					}
				}
			})

			Convey("It should build absolute distance bias when causal is false", func() {
				stateDict := state.NewDict().WithShape([]int{1, 3, 3})
				stateDict.Causal = false

				outputState, err := op.Forward(stateDict)

				So(err, ShouldBeNil)

				for index := range outputState.Out {
					So(outputState.Out[index], ShouldBeGreaterThanOrEqualTo, 0)
				}
			})
		})
	})
}

func BenchmarkALiBi_Forward(b *testing.B) {
	op := NewALiBi()
	shape := []int{32, 512, 512}

	for b.Loop() {
		_, _ = op.Forward(state.NewDict().WithShape(shape))
	}
}
