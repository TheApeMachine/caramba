package activation

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestGelu(t *testing.T) {
	Convey("Given a Gelu operation", t, func() {
		op := NewGelu()

		Convey("Forward", func() {
			Convey("It should return ~0 for large negative inputs", func() {
				out := op.Forward([]int{4}, []float64{-10, -8, -6, -5})
				for _, v := range out {
					So(v, ShouldAlmostEqual, 0, 1e-3)
				}
			})

			Convey("It should return ~x for large positive inputs", func() {
				out := op.Forward([]int{4}, []float64{5, 6, 8, 10})
				in := []float64{5, 6, 8, 10}
				for i, v := range out {
					So(v, ShouldAlmostEqual, in[i], 1e-3)
				}
			})

			Convey("It should return ~0 for zero input", func() {
				out := op.Forward([]int{4}, []float64{0, 0, 0, 0})
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
		op.Forward(shape, input)
	}
}
