package active_inference

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestFreeEnergy(t *testing.T) {
	Convey("Given a FreeEnergy operation", t, func() {
		op := NewFreeEnergy()

		Convey("Forward", func() {
			Convey("It should return zero for the standard normal (mu=0, log_sigma=0)", func() {
				// F = 0.5*(0 + 1 - 0 - 1) = 0
				n := 4
				mu := make([]float64, n)
				ls := make([]float64, n) // log_sigma = 0 → sigma = 1
				out := op.Forward([]int{n}, mu, ls)
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldAlmostEqual, 0.0, 1e-9)
			})

			Convey("It should be positive for non-standard beliefs", func() {
				mu := []float64{1.0, -1.0}
				ls := []float64{0.5, 0.5}
				out := op.Forward([]int{2}, mu, ls)
				So(out[0], ShouldBeGreaterThan, 0.0)
			})

			Convey("It should compute correctly for scalar case mu=1, log_sigma=0", func() {
				// F = 0.5*(1 + 1 - 0 - 1) = 0.5
				mu := []float64{1.0}
				ls := []float64{0.0}
				out := op.Forward([]int{1}, mu, ls)
				So(out[0], ShouldAlmostEqual, 0.5, 1e-9)
			})
		})
	})
}

func TestBeliefUpdate(t *testing.T) {
	Convey("Given a BeliefUpdate operation", t, func() {
		op := NewBeliefUpdate()

		Convey("Forward", func() {
			Convey("It should move mu towards zero when prediction error is zero", func() {
				// grad of F w.r.t. mu = mu + pred_err = mu when pred_err=0
				// so mu decreases proportionally
				mu := []float64{1.0, -2.0}
				ls := []float64{0.0, 0.0}
				pe := []float64{0.0, 0.0}
				out := op.Forward([]int{2, 1}, mu, ls, pe) // lr=1e-4
				So(out, ShouldHaveLength, 4)
				So(out[0], ShouldBeLessThan, mu[0]) // mu[0] decreased
				So(out[1], ShouldBeGreaterThan, mu[1]) // mu[1] increased (less negative)
			})

			Convey("It should return a vector of length 2N", func() {
				n := 8
				mu := make([]float64, n)
				ls := make([]float64, n)
				pe := make([]float64, n)
				out := op.Forward([]int{n, 1}, mu, ls, pe)
				So(out, ShouldHaveLength, 2*n)
			})

			Convey("It should not change beliefs when lr=0", func() {
				mu := []float64{1.5, -0.5}
				ls := []float64{0.3, -0.3}
				pe := []float64{0.1, 0.2}
				out := op.Forward([]int{2, 0}, mu, ls, pe) // lr = 0*1e-4 = 0
				So(out[0], ShouldAlmostEqual, mu[0], 1e-9)
				So(out[1], ShouldAlmostEqual, mu[1], 1e-9)
				So(out[2], ShouldAlmostEqual, ls[0], 1e-9)
				So(out[3], ShouldAlmostEqual, ls[1], 1e-9)
			})
		})
	})
}

func TestPrecisionWeight(t *testing.T) {
	Convey("Given a PrecisionWeight operation", t, func() {
		op := NewPrecisionWeight()

		Convey("Forward", func() {
			Convey("It should scale errors by exp(log_precision)", func() {
				// exp(0) = 1 → no scaling
				errVec := []float64{2.0, -3.0, 1.5}
				lp := []float64{0.0, 0.0, 0.0}
				out := op.Forward([]int{3}, errVec, lp)
				So(out[0], ShouldAlmostEqual, 2.0, 1e-9)
				So(out[1], ShouldAlmostEqual, -3.0, 1e-9)
				So(out[2], ShouldAlmostEqual, 1.5, 1e-9)
			})

			Convey("It should amplify errors when precision is high", func() {
				errVec := []float64{1.0}
				lp := []float64{math.Log(10.0)} // precision = 10
				out := op.Forward([]int{1}, errVec, lp)
				So(out[0], ShouldAlmostEqual, 10.0, 1e-6)
			})

			Convey("It should attenuate errors when precision is low", func() {
				errVec := []float64{1.0}
				lp := []float64{math.Log(0.5)} // precision = 0.5
				out := op.Forward([]int{1}, errVec, lp)
				So(out[0], ShouldAlmostEqual, 0.5, 1e-6)
			})
		})
	})
}

func TestExpectedFreeEnergy(t *testing.T) {
	Convey("Given an ExpectedFreeEnergy operation", t, func() {
		op := NewExpectedFreeEnergy()

		Convey("Forward", func() {
			Convey("It should return zero for a one-hot distribution (no ambiguity)", func() {
				// q[0,0]=1 → G[0] = -1*ln(1) = 0
				q := []float64{1.0, 0.0} // 1 state, 2 outcomes
				out := op.Forward([]int{1, 2}, q)
				So(out, ShouldHaveLength, 2)
				So(out[0], ShouldAlmostEqual, 0.0, 1e-6)
			})

			Convey("It should return the per-outcome entropy for a uniform distribution", func() {
				// For n=1, K=4: G[k] = -(1/K)*ln(1/K) = ln(K)/K
				K := 4
				n := 1
				q := make([]float64, n*K)
				for idx := range q {
					q[idx] = 1.0 / float64(K)
				}
				out := op.Forward([]int{n, K}, q)
				expected := math.Log(float64(K)) / float64(K)
				So(out[0], ShouldAlmostEqual, expected, 1e-6)
			})

			Convey("It should return G of length K", func() {
				n, K := 3, 5
				q := make([]float64, n*K)
				for idx := range q {
					q[idx] = 1.0 / float64(K)
				}
				out := op.Forward([]int{n, K}, q)
				So(out, ShouldHaveLength, K)
			})
		})
	})
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func BenchmarkFreeEnergy_Forward(b *testing.B) {
	op := NewFreeEnergy()
	n := 256
	mu := make([]float64, n)
	ls := make([]float64, n)

	for idx := range mu {
		mu[idx] = math.Sin(float64(idx)) * 0.5
		ls[idx] = math.Cos(float64(idx)) * 0.1
	}

	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		op.Forward([]int{n}, mu, ls)
	}
}

func BenchmarkBeliefUpdate_Forward(b *testing.B) {
	op := NewBeliefUpdate()
	n := 256
	mu := make([]float64, n)
	ls := make([]float64, n)
	pe := make([]float64, n)

	for idx := range mu {
		mu[idx] = math.Sin(float64(idx)) * 0.5
		ls[idx] = 0.0
		pe[idx] = math.Cos(float64(idx)) * 0.1
	}

	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		op.Forward([]int{n, 1}, mu, ls, pe)
	}
}

func BenchmarkPrecisionWeight_Forward(b *testing.B) {
	op := NewPrecisionWeight()
	n := 512
	errVec := make([]float64, n)
	lp := make([]float64, n)

	for idx := range errVec {
		errVec[idx] = math.Sin(float64(idx))
		lp[idx] = math.Cos(float64(idx)) * 2.0
	}

	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		op.Forward([]int{n}, errVec, lp)
	}
}

func BenchmarkExpectedFreeEnergy_Forward(b *testing.B) {
	op := NewExpectedFreeEnergy()
	n, K := 32, 16
	q := make([]float64, n*K)

	for idx := range q {
		q[idx] = 1.0 / float64(K)
	}

	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		op.Forward([]int{n, K}, q)
	}
}
