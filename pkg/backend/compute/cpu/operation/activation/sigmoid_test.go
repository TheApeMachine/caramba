package activation

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSigmoid(t *testing.T) {
	Convey("Given a Sigmoid operation", t, func() {
		op := NewSigmoid()

		Convey("Forward", func() {
			Convey("It should process odd-length tails", func() {
				out := op.Forward([]int{3}, []float64{-1, 0, 1})

				So(out[0], ShouldBeLessThan, out[1])
				So(out[1], ShouldAlmostEqual, 0.5, 1e-9)
				So(out[2], ShouldBeGreaterThan, out[1])
			})
		})
	})
}

func BenchmarkSigmoid_Forward(benchmark *testing.B) {
	op := NewSigmoid()
	input := make([]float64, 4097)

	for index := range input {
		input[index] = float64(index%512)/256 - 1
	}

	shape := []int{len(input)}

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		op.Forward(shape, input)
	}
}
