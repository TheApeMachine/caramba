package hebbian

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestHebbian_Step(t *testing.T) {
	Convey("Given a Hebbian optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should strengthen correlated weights", func() {
				opt := NewHebbian()
				stateDict := hebbianState([]float64{0.0}, []float64{1.0}, 0.1, 0)

				updated, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				So(updated.Out[0], ShouldAlmostEqual, 0.1)
			})

			Convey("It should clip weights when MaxNorm is set", func() {
				opt := NewHebbian()
				stateDict := hebbianState(
					[]float64{0.5, 0.5, 0.5, 0.5},
					[]float64{1.0, 1.0, 1.0, 1.0},
					1.0,
					1.0,
				)

				updated, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				norm := 0.0
				for _, v := range updated.Out {
					norm += v * v
				}
				So(stdmath.Sqrt(norm), ShouldBeLessThanOrEqualTo, 1.0+1e-9)
			})
		})
	})
}

func TestOjaRule_Step(t *testing.T) {
	Convey("Given an Oja rule optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should keep weights bounded via decay", func() {
				opt := NewOjaRule(0.01)
				params := make([]float64, 4)
				for idx := range params {
					params[idx] = 0.5
				}
				for range 5000 {
					// simulate unit post-synaptic activity
					grads := make([]float64, 4)
					for idx := range grads {
						grads[idx] = params[idx] // post*pre ≈ p
					}
					params = opt.Step(params, grads)
				}
				// weight norm should converge to ~1.0 (unit sphere)
				norm := 0.0
				for _, v := range params {
					norm += v * v
				}
				So(stdmath.Sqrt(norm), ShouldAlmostEqual, 1.0, 0.1)
			})
		})
	})
}

func BenchmarkHebbian_Step(b *testing.B) {
	opt := NewHebbian()
	n := 1 << 20
	params := make([]float64, n)
	grads := make([]float64, n)
	for i := range params {
		params[i] = float64(i) * 1e-6
		grads[i] = float64(i%2*2-1) * 1e-4
	}
	stateDict := hebbianState(params, grads, 0.01, 1.0)
	b.ResetTimer()
	for b.Loop() {
		updated, err := opt.Step(stateDict)
		if err != nil {
			b.Fatalf("Step failed: %v", err)
		}
		stateDict.WithParams(updated.Out)
	}
}

func hebbianState(params, grads []float64, lr, maxNorm float64) *state.Dict {
	return state.NewDict().
		WithLR(lr).
		WithMaxNorm(maxNorm).
		WithParams(params).
		WithGrads(grads)
}
