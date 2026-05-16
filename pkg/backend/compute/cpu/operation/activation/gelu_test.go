package activation

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestGelu(t *testing.T) {
	Convey("Given a Gelu operation", t, func() {
		op := NewGelu()

		Convey("Forward", func() {
			Convey("It should compute tanh-form GELU", func() {
				input := []float64{-3, -1, 0, 1, 3}
				out := forwardGelu(op, input)

				for index, value := range input {
					expected := geluReference(value)
					So(out[index], ShouldAlmostEqual, expected, 1e-12)
				}
			})

			Convey("It should return ~0 for large negative inputs", func() {
				out := forwardGelu(op, []float64{-10, -8, -6, -5})
				for _, v := range out {
					So(v, ShouldAlmostEqual, 0, 1e-3)
				}
			})

			Convey("It should return ~x for large positive inputs", func() {
				in := []float64{5, 6, 8, 10}
				out := forwardGelu(op, in)
				for i, v := range out {
					So(v, ShouldAlmostEqual, in[i], 1e-3)
				}
			})

			Convey("It should return ~0 for zero input", func() {
				out := forwardGelu(op, []float64{0, 0, 0, 0})
				for _, v := range out {
					So(math.Abs(v), ShouldBeLessThan, 1e-9)
				}
			})
		})
	})
}

func BenchmarkGelu_Forward(b *testing.B) {
	op := NewGelu()
	input := make([]float64, 4096)
	for i := range input {
		input[i] = float64(i%512)/256 - 1
	}
	shape := []int{4096}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		stateDict := state.NewDict().WithShape(shape).WithInput(input)
		_, _ = op.Forward(stateDict)
	}
}

func forwardGelu(op *Gelu, input []float64) []float64 {
	stateDict := state.NewDict().
		WithShape([]int{len(input)}).
		WithInput(input)

	out, err := op.Forward(stateDict)

	So(err, ShouldBeNil)

	return out.Out
}

func geluReference(value float64) float64 {
	cube := value * value * value
	inner := 0.7978845608028654 * (value + 0.044715*cube)

	return 0.5 * value * (1 + math.Tanh(inner))
}
