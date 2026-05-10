package attention

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
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
				out := op.Forward(shape, q, k, v)
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
				out := op.Forward(shape, q, k, v)
				for _, val := range out {
					So(math.Abs(val-1.0), ShouldBeLessThan, 1e-9)
				}
			})
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

	for i := 0; i < b.N; i++ {
		op.Forward(shape, q, k, v)
	}
}
