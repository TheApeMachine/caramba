package lion

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLion_Step(t *testing.T) {
	Convey("Given a Lion optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should reduce a quadratic loss", func() {
				opt := NewLion()
				stateDict := lionState([]float64{2.0}, []float64{4.0}, 1e-3, 0.9, 0.99, 0)

				for range 5000 {
					updated, err := opt.Step(stateDict)
					So(err, ShouldBeNil)

					stateDict.
						WithParams(updated.Out).
						WithGrads([]float64{2 * updated.Out[0]})
				}
				So(stdmath.Abs(stateDict.Out[0]), ShouldBeLessThan, 0.5)
			})

			Convey("It should produce unit-magnitude updates", func() {
				opt := NewLion()
				params := []float64{1.0, -1.0, 0.5}
				grads := []float64{0.3, -0.7, 1.2}
				stateDict := lionState(params, grads, 0.01, 0.9, 0.99, 0)

				updated, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				for idx := range updated.Out {
					diff := stdmath.Abs(updated.Out[idx] - params[idx])
					So(diff, ShouldAlmostEqual, 0.01, 1e-9)
				}
			})
		})
	})
}

func BenchmarkLion_Step(b *testing.B) {
	opt := NewLion()
	n := 1 << 20
	params := make([]float64, n)
	grads := make([]float64, n)
	for i := range params {
		params[i] = 1e-3
		grads[i] = float64(i%3-1) * 1e-4
	}
	stateDict := lionState(params, grads, 1e-4, 0.9, 0.99, 0.01)
	b.ResetTimer()
	for b.Loop() {
		updated, err := opt.Step(stateDict)
		if err != nil {
			b.Fatalf("Step failed: %v", err)
		}
		stateDict.WithParams(updated.Out)
	}
}

func lionState(params, grads []float64, lr, beta1, beta2, wd float64) *state.Dict {
	return state.NewDict().
		WithLR(lr).
		WithBeta1(beta1).
		WithBeta2(beta2).
		WithWD(wd).
		WithParams(params).
		WithGrads(grads)
}
