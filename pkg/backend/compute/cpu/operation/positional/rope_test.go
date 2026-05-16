package positional

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestRoPE(t *testing.T) {
	Convey("Given a RoPE operation", t, func() {
		op := NewRoPE(10000.0, 8)

		Convey("Forward", func() {
			Convey("It should preserve vector norms (rotation is isometric)", func() {
				batch, heads, seq, dim := 1, 1, 4, 8
				n := batch * heads * seq * dim
				in := make([]float64, n)
				for i := range in {
					in[i] = float64(i+1) / float64(n)
				}
				outputState, err := op.Forward(
					state.NewDict().
						WithShape([]int{batch, heads, seq, dim}).
						WithInput(in),
				)
				So(err, ShouldBeNil)
				out := outputState.Out
				So(out, ShouldHaveLength, n)

				// Compare norm per position — rotation must preserve length.
				for pos := 0; pos < seq; pos++ {
					var normIn, normOut float64
					for d := 0; d < dim; d++ {
						idx := pos*dim + d
						normIn += in[idx] * in[idx]
						normOut += out[idx] * out[idx]
					}
					So(math.Abs(math.Sqrt(normOut)-math.Sqrt(normIn)), ShouldBeLessThan, 1e-9)
				}
			})

			Convey("It should return a different encoding for each position", func() {
				batch, heads, seq, dim := 1, 1, 2, 8
				in := make([]float64, batch*heads*seq*dim)
				for i := range in {
					in[i] = 1.0
				}
				outputState, err := op.Forward(
					state.NewDict().
						WithShape([]int{batch, heads, seq, dim}).
						WithInput(in),
				)
				So(err, ShouldBeNil)
				out := outputState.Out
				pos0 := out[:dim]
				pos1 := out[dim : 2*dim]
				same := true
				for i := range pos0 {
					if math.Abs(pos0[i]-pos1[i]) > 1e-12 {
						same = false
						break
					}
				}
				So(same, ShouldBeFalse)
			})

			Convey("It should honor absolute decode position offsets", func() {
				in := []float64{1, 1, 1, 1}
				baseState := state.NewDict().
					WithShape([]int{1, 1, 1, 4}).
					WithInput(in)
				offsetState := state.NewDict().
					WithShape([]int{1, 1, 1, 4}).
					WithInput(in)
				offsetState.PositionStart = 3

				baseOutput, err := op.Forward(baseState)
				So(err, ShouldBeNil)
				offsetOutput, err := op.Forward(offsetState)
				So(err, ShouldBeNil)

				So(offsetOutput.Out, ShouldNotResemble, baseOutput.Out)
			})

			Convey("It should rotate split-half RoPE pairs when configured", func() {
				input := []float64{1, 2, 3, 4}
				operationState := state.NewDict().
					WithShape([]int{1, 1, 1, 4}).
					WithInput(input)
				operationState.Mode = "half"
				operationState.PositionStart = 1

				outputState, err := op.Forward(operationState)

				So(err, ShouldBeNil)

				cos0 := math.Cos(1)
				sin0 := math.Sin(1)
				cos1 := math.Cos(0.01)
				sin1 := math.Sin(0.01)
				expected := []float64{
					1*cos0 - 3*sin0,
					2*cos1 - 4*sin1,
					1*sin0 + 3*cos0,
					2*sin1 + 4*cos1,
				}

				for index := range expected {
					So(math.Abs(outputState.Out[index]-expected[index]), ShouldBeLessThan, 1e-9)
				}
			})

			Convey("It should apply Llama 3 scaled split-half frequencies", func() {
				input := []float64{1, 2, 3, 4}
				operationState := state.NewDict().
					WithShape([]int{1, 1, 1, 4}).
					WithInput(input)
				operationState.Mode = "half"
				operationState.PositionStart = 1
				operationState.Base = 1e8
				operationState.RoPEType = "llama3"
				operationState.RoPEFactor = 2
				operationState.RoPELowFreqFactor = 1
				operationState.RoPEHighFreqFactor = 4
				operationState.RoPEOriginalContext = 8192

				outputState, err := op.Forward(operationState)

				So(err, ShouldBeNil)

				cos0 := math.Cos(1)
				sin0 := math.Sin(1)
				cos1 := math.Cos(0.00005)
				sin1 := math.Sin(0.00005)
				expected := []float64{
					1*cos0 - 3*sin0,
					2*cos1 - 4*sin1,
					1*sin0 + 3*cos0,
					2*sin1 + 4*cos1,
				}

				for index := range expected {
					So(math.Abs(outputState.Out[index]-expected[index]), ShouldBeLessThan, 1e-9)
				}
			})

			Convey("It should reject projection tensors before head shaping", func() {
				_, err := op.Forward(
					state.NewDict().
						WithShape([]int{1, 2, 8}).
						WithInput(make([]float64, 16)),
				)

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "expected [batch, num_heads, seq_len, head_dim]")
			})
		})
	})
}

func BenchmarkRoPE_Forward(b *testing.B) {
	op := NewRoPE(10000.0, 64)
	batch, heads, seq, dim := 1, 8, 512, 64
	n := batch * heads * seq * dim
	in := make([]float64, n)
	for i := range in {
		in[i] = float64(i) / float64(n)
	}
	shape := []int{batch, heads, seq, dim}

	b.ResetTimer()

	for b.Loop() {
		_, _ = op.Forward(state.NewDict().WithShape(shape).WithInput(in))
	}
}
