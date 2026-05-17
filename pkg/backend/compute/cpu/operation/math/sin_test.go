package math

import (
	gomath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestSin_Forward(t *testing.T) {
	Convey("Given a Sin operation", t, func() {
		op := NewSin()

		Convey("It should match math.Sin at canonical points", func() {
			inputs := []float64{
				0, 1, -1, 0.5, -0.5,
				gomath.Pi / 6, gomath.Pi / 4, gomath.Pi / 3, gomath.Pi / 2,
				gomath.Pi, -gomath.Pi, 2 * gomath.Pi, -2 * gomath.Pi,
				1e-9, -1e-9, 1e-3, -1e-3,
			}
			out := forwardMath(op, []int{len(inputs)}, inputs)
			for index, value := range inputs {
				expected := gomath.Sin(value)
				abs := gomath.Abs(out[index] - expected)
				So(abs, ShouldBeLessThanOrEqualTo, 4e-15)
			}
		})

		// Backend-kernel parity at the prescribed sizes (1, 7, 64, 1024, 8192)
		// exercises single-lane, odd-tail, single-vector, and multi-vector paths.
		for _, n := range []int{1, 7, 64, 1024, 8192} {
			Convey("It should match math.Sin within tight ULPs at N="+itoa(n), func() {
				inputs := make([]float64, n)
				for index := range inputs {
					inputs[index] = float64(index)*0.137 - 4.71
				}
				out := forwardMath(op, []int{n}, inputs)
				for index, value := range inputs {
					expected := gomath.Sin(value)
					ulp := gomath.Nextafter(expected, gomath.Inf(1)) - expected
					if ulp == 0 {
						ulp = gomath.Nextafter(1.0, gomath.Inf(1)) - 1.0
					}
					diff := gomath.Abs(out[index] - expected)
					So(diff, ShouldBeLessThanOrEqualTo, 4*gomath.Abs(ulp))
				}
			})
		}

		Convey("It should return an error for a nil input buffer", func() {
			stateDict := state.NewDict().
				WithShape([]int{0}).
				WithInputs([]float64(nil))

			_, err := op.Forward(stateDict)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "stateDict.Inputs[0] must be non-nil")
		})
	})
}

func BenchmarkSin_Forward(b *testing.B) {
	op := NewSin()
	data := make([]float64, 1024)
	for index := range data {
		data[index] = float64(index)*0.01 - 5.0
	}
	for b.Loop() {
		stateDict := state.NewDict().WithShape([]int{1024}).WithInput(data)
		_, _ = op.Forward(stateDict)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	negative := n < 0
	if negative {
		n = -n
	}
	var buffer [20]byte
	cursor := len(buffer)
	for n > 0 {
		cursor--
		buffer[cursor] = byte('0' + n%10)
		n /= 10
	}
	if negative {
		cursor--
		buffer[cursor] = '-'
	}
	return string(buffer[cursor:])
}
