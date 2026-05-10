package adam

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestAdam_Step(t *testing.T) {
	Convey("Given an Adam optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should converge a quadratic loss to zero", func() {
				// f(p) = p², grad = 2p; Adam should reach ~0 in 2000 steps
				opt := NewAdam(0.01, 0.9, 0.999, 1e-8, 0)
				params := []float64{5.0}
				for range 2000 {
					grads := []float64{2 * params[0]}
					params = opt.Step(params, grads)
				}
				So(stdmath.Abs(params[0]), ShouldBeLessThan, 0.1)
			})

			Convey("It should apply decoupled weight decay in AdamW mode", func() {
				opt := NewAdamW(0.01, 0.9, 0.999, 1e-8, 0.1)
				params := []float64{1.0, 1.0}
				grads := []float64{0.0, 0.0}
				out := opt.Step(params, grads)
				// with zero grad, only weight decay acts: p -= lr*wd*p
				So(out[0], ShouldBeLessThan, params[0])
			})

			Convey("It should not mutate input slices", func() {
				opt := NewAdam(0.01, 0.9, 0.999, 1e-8, 0)
				params := []float64{3.0, 3.0}
				grads := []float64{1.0, 1.0}
				_ = opt.Step(params, grads)
				So(params[0], ShouldEqual, 3.0)
			})
		})
	})
}

func TestAdaMax_Step(t *testing.T) {
	Convey("Given an AdaMax optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should reduce a quadratic loss", func() {
				opt := NewAdaMax(0.002, 0.9, 0.999, 1e-8)
				params := []float64{3.0}
				for range 2000 {
					grads := []float64{2 * params[0]}
					params = opt.Step(params, grads)
				}
				So(stdmath.Abs(params[0]), ShouldBeLessThan, 0.5)
			})
		})
	})
}

func BenchmarkAdam_Step(b *testing.B) {
	opt := NewAdam(0.001, 0.9, 0.999, 1e-8, 0)
	n := 1 << 20
	params := make([]float64, n)
	grads := make([]float64, n)
	for i := range params {
		params[i] = 1e-3
		grads[i] = 1e-4
	}
	b.ResetTimer()
	for range b.N {
		params = opt.Step(params, grads)
	}
}
