package sgd

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestSGD_Step(t *testing.T) {
	Convey("Given an SGD optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should move parameters in the negative gradient direction", func() {
				opt := NewSGD()
				stateDict := sgdState([]float64{1.0, 2.0, 3.0}, []float64{1.0, 1.0, 1.0}, 0.1, 0, 0, false)

				updated, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				So(updated.Out[0], ShouldAlmostEqual, 0.9)
				So(updated.Out[1], ShouldAlmostEqual, 1.9)
				So(updated.Out[2], ShouldAlmostEqual, 2.9)
			})

			Convey("It should accumulate momentum across steps", func() {
				opt := NewSGD()
				params := []float64{1.0, 1.0}
				grads := []float64{1.0, 1.0}
				stateDict := sgdState(params, grads, 0.1, 0.9, 0, false)
				p1, err := opt.Step(stateDict)
				So(err, ShouldBeNil)
				firstOut := append([]float64(nil), p1.Out...)
				stateDict.WithParams(firstOut).WithGrads(grads)

				p2, err := opt.Step(stateDict)
				So(err, ShouldBeNil)

				delta1 := params[0] - firstOut[0]
				delta2 := firstOut[0] - p2.Out[0]
				So(delta2, ShouldBeGreaterThan, delta1)
			})

			Convey("It should apply weight decay", func() {
				opt := NewSGD()
				stateDict := sgdState([]float64{2.0}, []float64{0.0}, 0.1, 0, 0.1, false)

				updated, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				So(updated.Out[0], ShouldAlmostEqual, 1.98)
			})

			Convey("It should not mutate the input params slice", func() {
				opt := NewSGD()
				params := []float64{5.0, 5.0}
				stateDict := sgdState(params, []float64{1.0, 1.0}, 0.1, 0, 0, false)
				_, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				So(params[0], ShouldEqual, 5.0)
			})
		})
	})
}

func BenchmarkSGD_Step(b *testing.B) {
	opt := NewSGD()
	n := 1 << 20
	params := make([]float64, n)
	grads := make([]float64, n)
	for i := range params {
		params[i] = float64(i) * 1e-4
		grads[i] = float64(i) * 1e-5
	}
	stateDict := sgdState(params, grads, 0.01, 0.9, 1e-4, false)
	b.ResetTimer()
	for b.Loop() {
		updated, err := opt.Step(stateDict)
		if err != nil {
			b.Fatalf("Step failed: %v", err)
		}
		stateDict.WithParams(updated.Out)
	}
}

func sgdState(params, grads []float64, lr, momentum, wd float64, nesterov bool) *state.Dict {
	return state.NewDict().
		WithLR(lr).
		WithMomentum(momentum).
		WithWD(wd).
		WithNesterov(nesterov).
		WithParams(params).
		WithGrads(grads)
}
