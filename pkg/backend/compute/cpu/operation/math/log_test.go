package math

import (
	gomath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestLog_Forward(t *testing.T) {
	Convey("Given a Log operation", t, func() {
		op := NewLog()

		Convey("Forward", func() {
			Convey("It should match math.Log for sample positives", func() {
				inputs := []float64{1.0, 2.0, 0.5, gomath.E, 10.0, 100.0, 1e6, 1e-6, 1e10}
				out := op.Forward([]int{len(inputs)}, inputs)
				for index, value := range inputs {
					expected := gomath.Log(value)
					absErr := gomath.Abs(out[index] - expected)
					relErr := absErr / (gomath.Abs(expected) + 1e-12)
					So(relErr, ShouldBeLessThan, 1e-7)
				}
			})

			Convey("It should match math.Log across many lanes", func() {
				inputs := make([]float64, 257)
				for index := range inputs {
					inputs[index] = float64(index)*0.5 + 0.001
				}
				out := op.Forward([]int{len(inputs)}, inputs)
				for index, value := range inputs {
					expected := gomath.Log(value)
					absErr := gomath.Abs(out[index] - expected)
					relErr := absErr / (gomath.Abs(expected) + 1e-12)
					So(relErr, ShouldBeLessThan, 1e-7)
				}
			})
		})
	})
}

func BenchmarkLog_Forward(b *testing.B) {
	op := NewLog()
	data := make([]float64, 1024)
	for index := range data {
		data[index] = float64(index)*0.5 + 0.001
	}
	b.ResetTimer()
	for repeat := 0; repeat < b.N; repeat++ {
		op.Forward([]int{1024}, data)
	}
}
