package positional

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
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
				out := op.Forward([]int{batch, heads, seq, dim}, in)
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
				out := op.Forward([]int{batch, heads, seq, dim}, in)
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

	for i := 0; i < b.N; i++ {
		op.Forward(shape, in)
	}
}
