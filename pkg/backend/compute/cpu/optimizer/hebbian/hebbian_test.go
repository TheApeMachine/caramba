package hebbian

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestHebbian_Step(t *testing.T) {
	Convey("Given a Hebbian optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should strengthen correlated weights", func() {
				opt := NewHebbian(0.1, 0)
				params := []float64{0.0}
				grads := []float64{1.0}
				out := opt.Step(params, grads)
				So(out[0], ShouldAlmostEqual, 0.1)
			})

			Convey("It should clip weights when MaxNorm is set", func() {
				opt := NewHebbian(1.0, 1.0)
				params := []float64{0.5, 0.5, 0.5, 0.5}
				grads := []float64{1.0, 1.0, 1.0, 1.0}
				out := opt.Step(params, grads)
				norm := 0.0
				for _, v := range out {
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
	opt := NewHebbian(0.01, 1.0)
	n := 1 << 20
	params := make([]float64, n)
	grads := make([]float64, n)
	for i := range params {
		params[i] = float64(i) * 1e-6
		grads[i] = float64(i%2*2-1) * 1e-4
	}
	b.ResetTimer()
	for range b.N {
		params = opt.Step(params, grads)
	}
}
