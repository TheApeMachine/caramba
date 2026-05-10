package activation

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestReLU(t *testing.T) {
	Convey("Given a ReLU operation", t, func() {
		op := NewReLU()

		Convey("Forward", func() {
			Convey("It should zero negative values and pass positive values through", func() {
				out := op.Forward([]int{8}, []float64{-4, -1, 0, 1, 2, -0.5, 3, -3})
				So(out, ShouldResemble, []float64{0, 0, 0, 1, 2, 0, 3, 0})
			})

			Convey("It should return an empty slice for empty input", func() {
				out := op.Forward([]int{0})
				So(out, ShouldHaveLength, 0)
			})

			Convey("It should not modify values that are already non-negative", func() {
				in := []float64{0, 1, 2, 3, 4, 5, 6, 7}
				out := op.Forward([]int{8}, in)
				So(out, ShouldResemble, in)
			})
		})
	})
}

func BenchmarkReLU_Forward(b *testing.B) {
	op := NewReLU()
	input := make([]float64, 4096)
	for i := range input {
		input[i] = float64(i%512) - 256
	}
	shape := []int{4096}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward(shape, input)
	}
}
