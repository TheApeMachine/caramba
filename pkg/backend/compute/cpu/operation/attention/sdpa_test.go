package attention

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestSDPA(t *testing.T) {
	Convey("Given an SDPA operation", t, func() {
		op := NewSDPA()

		Convey("Forward", func() {
			Convey("It should produce output of correct shape", func() {
				batch, heads, seq, dim := 1, 2, 4, 8
				n := batch * heads * seq * dim
				q := make([]float64, n)
				k := make([]float64, n)
				v := make([]float64, n)
				for i := range q {
					q[i] = float64(i%dim) / float64(dim)
					k[i] = float64(i%dim) / float64(dim)
					v[i] = 1.0
				}
				shape := []int{batch, heads, seq, dim}
				outputState, err := op.Forward(
					state.NewDict().
						WithShape(shape).
						WithInputs(q, k, v),
				)
				So(err, ShouldBeNil)
				out := outputState.Out
				So(out, ShouldHaveLength, n)
			})

			Convey("It should produce uniform attention weights for identical Q and K rows", func() {
				// When all K rows are identical, softmax weights are uniform,
				// so output should equal v (all ones) for each query.
				batch, heads, seq, dim := 1, 1, 4, 4
				n := batch * heads * seq * dim
				q := make([]float64, n)
				k := make([]float64, n)
				v := make([]float64, n)
				for i := range v {
					v[i] = 1.0
				}
				shape := []int{batch, heads, seq, dim}
				outputState, err := op.Forward(
					state.NewDict().
						WithShape(shape).
						WithInputs(q, k, v),
				)
				So(err, ShouldBeNil)
				out := outputState.Out
				for _, val := range out {
					So(math.Abs(val-1.0), ShouldBeLessThan, 1e-9)
				}
			})

			Convey("It should honor causal masking", func() {
				shape := []int{1, 1, 3, 1}
				q := []float64{0, 0, 0}
				k := []float64{0, 0, 0}
				v := []float64{10, 20, 30}

				stateDict := state.NewDict().
					WithShape(shape).
					WithInputs(q, k, v)
				stateDict.Causal = true
				outputState, err := op.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{10, 15, 20})
			})

			Convey("It should honor causal masking with cached K/V longer than Q", func() {
				shape := []int{1, 1, 1, 1}
				q := []float64{0}
				k := []float64{0, 0, 0}
				v := []float64{10, 20, 30}

				stateDict := state.NewDict().
					WithShape(shape).
					WithInputs(q, k, v)
				stateDict.Causal = true
				outputState, err := op.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{20})
			})
		})
	})
}

func TestAttentionRowScoresKernel(t *testing.T) {
	Convey("Given an attention row score kernel", t, func() {
		Convey("It should compute scaled dot products", func() {
			scores := make([]float64, 2)
			query := []float64{1, 2, 3, 4}
			keys := []float64{
				1, 0, 0, 0,
				0, 1, 0, 1,
			}

			attentionRowScoresKernel(scores, query, keys, 2, 4, 0.5)

			So(scores, ShouldResemble, []float64{0.5, 3.0})
		})
	})
}

func TestAttentionRowOutputKernel(t *testing.T) {
	Convey("Given an attention row output kernel", t, func() {
		Convey("It should compute weighted value accumulation", func() {
			output := make([]float64, 4)
			scores := []float64{0.25, 0.75}
			values := []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
			}

			attentionRowOutputKernel(output, scores, values, 2, 4)

			So(output, ShouldResemble, []float64{4, 5, 6, 7})
		})
	})
}

func BenchmarkSDPA_Forward(b *testing.B) {
	op := NewSDPA()
	batch, heads, seq, dim := 1, 8, 64, 64
	n := batch * heads * seq * dim
	q := make([]float64, n)
	k := make([]float64, n)
	v := make([]float64, n)
	for i := range q {
		q[i] = float64(i%dim) / float64(dim)
		k[i] = q[i]
		v[i] = 1.0
	}
	shape := []int{batch, heads, seq, dim}

	b.ResetTimer()

	for b.Loop() {
		_, _ = op.Forward(
			state.NewDict().
				WithShape(shape).
				WithInputs(q, k, v),
		)
	}
}
