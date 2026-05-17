package math

import (
	gomath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestCos_Forward(t *testing.T) {
	Convey("Given a Cos operation", t, func() {
		op := NewCos()

		Convey("It should match math.Cos at canonical points", func() {
			inputs := []float64{
				0, 1, -1, 0.5, -0.5,
				gomath.Pi / 6, gomath.Pi / 4, gomath.Pi / 3, gomath.Pi / 2,
				gomath.Pi, -gomath.Pi, 2 * gomath.Pi, -2 * gomath.Pi,
				1e-9, -1e-9, 1e-3, -1e-3,
			}
			out := forwardMath(op, []int{len(inputs)}, inputs)
			for index, value := range inputs {
				expected := gomath.Cos(value)
				abs := gomath.Abs(out[index] - expected)
				So(abs, ShouldBeLessThanOrEqualTo, 4e-15)
			}
		})

		for _, n := range []int{1, 7, 64, 1024, 8192} {
			Convey("It should match math.Cos within tight ULPs at N="+itoa(n), func() {
				inputs := make([]float64, n)
				for index := range inputs {
					inputs[index] = float64(index)*0.137 - 4.71
				}
				out := forwardMath(op, []int{n}, inputs)
				for index, value := range inputs {
					expected := gomath.Cos(value)
					ulp := gomath.Nextafter(expected, gomath.Inf(1)) - expected
					if ulp == 0 {
						ulp = gomath.Nextafter(1.0, gomath.Inf(1)) - 1.0
					}
					diff := gomath.Abs(out[index] - expected)
					So(diff, ShouldBeLessThanOrEqualTo, 4*gomath.Abs(ulp))
				}
			})
		}

		Convey("It should return an error when stateDict.Out is too small", func() {
			stateDict := state.NewDict().
				WithShape([]int{3}).
				WithInput([]float64{0, 1, 2}).
				WithOut(make([]float64, 2))

			_, err := op.Forward(stateDict)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "stateDict.Out length 2")
			So(err.Error(), ShouldContainSubstring, "before cosKernel")
		})
	})
}

func BenchmarkCos_Forward(b *testing.B) {
	op := NewCos()
	data := make([]float64, 1024)
	for index := range data {
		data[index] = float64(index)*0.01 - 5.0
	}

	stateDict := state.NewDict().
		WithShape([]int{1024}).
		WithInput(data).
		WithOut(make([]float64, len(data)))

	b.ResetTimer()

	for b.Loop() {
		_, _ = op.Forward(stateDict)
	}
}
