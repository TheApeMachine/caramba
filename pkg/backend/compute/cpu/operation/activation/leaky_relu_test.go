package activation

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestLeakyReLU(t *testing.T) {
	Convey("Given a LeakyReLU operation", t, func() {
		op := NewLeakyReLU(0.25)

		Convey("Forward", func() {
			Convey("It should process odd-length tails", func() {
				out := op.Forward([]int{3}, []float64{-2, 0, 4})

				So(out, ShouldResemble, []float64{-0.5, 0, 4})
			})
		})
	})
}

func BenchmarkLeakyReLU_Forward(benchmark *testing.B) {
	op := NewLeakyReLU(0.01)
	input := make([]float64, 4097)

	for index := range input {
		input[index] = float64(index%512) - 256
	}

	shape := []int{len(input)}

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		op.Forward(shape, input)
	}
}
