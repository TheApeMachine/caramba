package lion

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestLion_Step(t *testing.T) {
	Convey("Given a Lion optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should reduce a quadratic loss", func() {
				opt := NewLion(1e-3, 0.9, 0.99, 0)
				params := []float64{2.0}
				for range 5000 {
					grads := []float64{2 * params[0]}
					params = opt.Step(params, grads)
				}
				So(stdmath.Abs(params[0]), ShouldBeLessThan, 0.5)
			})

			Convey("It should produce unit-magnitude updates", func() {
				opt := NewLion(0.01, 0.9, 0.99, 0)
				params := []float64{1.0, -1.0, 0.5}
				grads := []float64{0.3, -0.7, 1.2}
				out := opt.Step(params, grads)
				// each param changes by exactly lr (sign update)
				for idx := range out {
					diff := stdmath.Abs(out[idx] - params[idx])
					So(diff, ShouldAlmostEqual, 0.01, 1e-9)
				}
			})
		})
	})
}

func BenchmarkLion_Step(b *testing.B) {
	opt := NewLion(1e-4, 0.9, 0.99, 0.01)
	n := 1 << 20
	params := make([]float64, n)
	grads := make([]float64, n)
	for i := range params {
		params[i] = 1e-3
		grads[i] = float64(i%3-1) * 1e-4
	}
	b.ResetTimer()
	for range b.N {
		params = opt.Step(params, grads)
	}
}
