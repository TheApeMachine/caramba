package sgd

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSGD_Step(t *testing.T) {
	Convey("Given an SGD optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should move parameters in the negative gradient direction", func() {
				opt := NewSGD(0.1, 0, 0, false)
				params := []float64{1.0, 2.0, 3.0}
				grads := []float64{1.0, 1.0, 1.0}
				out := opt.Step(params, grads)
				So(out[0], ShouldAlmostEqual, 0.9)
				So(out[1], ShouldAlmostEqual, 1.9)
				So(out[2], ShouldAlmostEqual, 2.9)
			})

			Convey("It should accumulate momentum across steps", func() {
				opt := NewSGD(0.1, 0.9, 0, false)
				params := []float64{1.0, 1.0}
				grads := []float64{1.0, 1.0}
				p1 := opt.Step(params, grads)
				p2 := opt.Step(p1, grads)
				// second step has larger update due to momentum
				delta1 := params[0] - p1[0]
				delta2 := p1[0] - p2[0]
				So(delta2, ShouldBeGreaterThan, delta1)
			})

			Convey("It should apply weight decay", func() {
				opt := NewSGD(0.1, 0, 0.1, false)
				params := []float64{2.0}
				grads := []float64{0.0}
				out := opt.Step(params, grads)
				// p -= lr * wd * p = 2 - 0.1*0.1*2 = 1.98
				So(out[0], ShouldAlmostEqual, 1.98)
			})

			Convey("It should not mutate the input params slice", func() {
				opt := NewSGD(0.1, 0, 0, false)
				params := []float64{5.0, 5.0}
				grads := []float64{1.0, 1.0}
				_ = opt.Step(params, grads)
				So(params[0], ShouldEqual, 5.0)
			})
		})
	})
}

func BenchmarkSGD_Step(b *testing.B) {
	opt := NewSGD(0.01, 0.9, 1e-4, false)
	n := 1 << 20
	params := make([]float64, n)
	grads := make([]float64, n)
	for i := range params {
		params[i] = float64(i) * 1e-4
		grads[i] = float64(i) * 1e-5
	}
	b.ResetTimer()
	for range b.N {
		params = opt.Step(params, grads)
	}
}
