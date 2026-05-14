package active_inference

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type stateOperation interface {
	Forward(*state.Dict) (*state.Dict, error)
}

func forwardActive(
	operation stateOperation, shape []int, inputs ...[]float64,
) []float64 {
	stateDict := state.NewDict().WithShape(shape)
	values := make([]any, len(inputs))

	for index := range inputs {
		values[index] = inputs[index]
	}

	stateDict.WithInputs(values...)
	outputState, err := operation.Forward(stateDict)

	So(err, ShouldBeNil)

	return outputState.Out
}

func assertAllFiniteFloats(vals []float64) {
	for _, v := range vals {
		So(math.IsNaN(v), ShouldBeFalse)
		So(math.IsInf(v, 0), ShouldBeFalse)
	}
}

func TestFreeEnergy(t *testing.T) {
	Convey("Given a FreeEnergy operation", t, func() {
		op := NewFreeEnergy()

		Convey("Forward", func() {
			Convey("It should return zero for the standard normal (mu=0, log_sigma=0)", func() {
				// F = 0.5*(0 + 1 - 0 - 1) = 0
				n := 4
				mu := make([]float64, n)
				ls := make([]float64, n) // log_var = 0 → variance = 1
				out := forwardActive(op, []int{n}, mu, ls)
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldAlmostEqual, 0.0, 1e-9)
			})

			Convey("It should be positive for non-standard beliefs", func() {
				mu := []float64{1.0, -1.0}
				ls := []float64{0.5, 0.5}
				out := forwardActive(op, []int{2}, mu, ls)
				So(out[0], ShouldBeGreaterThan, 0.0)
			})

			Convey("It should compute correctly for scalar case mu=1, log_var=0", func() {
				// F = 0.5*(1 + 1 - 0 - 1) = 0.5
				mu := []float64{1.0}
				ls := []float64{0.0}
				out := forwardActive(op, []int{1}, mu, ls)
				So(out[0], ShouldAlmostEqual, 0.5, 1e-9)
			})

			Convey("It should return scalar zero for empty (N=0) inputs", func() {
				out := forwardActive(op, []int{0}, []float64{}, []float64{})
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldAlmostEqual, 0.0, 1e-15)
			})

			Convey("It should error when mu and log_var lengths disagree with N", func() {
				stateDict := state.NewDict().
					WithShape([]int{2}).
					WithInputs([]float64{1}, []float64{0, 0})
				_, err := op.Forward(stateDict)

				So(err, ShouldNotBeNil)
			})

			Convey("It should propagate NaN from inputs", func() {
				out := forwardActive(op, []int{1}, []float64{math.NaN()}, []float64{0})
				So(math.IsNaN(out[0]), ShouldBeTrue)
			})

			Convey("It should propagate signed infinities from inputs", func() {
				outPos := forwardActive(op, []int{1}, []float64{math.Inf(1)}, []float64{0.0})
				So(math.IsInf(outPos[0], 1), ShouldBeTrue)

				outNeg := forwardActive(op, []int{1}, []float64{0.0}, []float64{math.Inf(-1)})
				So(math.IsNaN(outNeg[0]) || math.IsInf(outNeg[0], 0), ShouldBeTrue)
			})

			Convey("It should yield finite output for large-but-representable log-variance", func() {
				mu := []float64{10.0, -10.0}
				ls := []float64{50.0, 50.0}
				out := forwardActive(op, []int{2}, mu, ls)
				assertAllFiniteFloats(out)
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
				out := forwardActive(op, []int{2, 1}, mu, ls, pe) // lr=1e-4
				So(out, ShouldHaveLength, 4)
				So(out[0], ShouldBeLessThan, mu[0])    // mu[0] decreased
				So(out[1], ShouldBeGreaterThan, mu[1]) // mu[1] increased (less negative)
			})

			Convey("It should match analytic update for N=1", func() {
				mu := []float64{2.0}
				ls := []float64{0.0}
				pe := []float64{1.0}
				lrStep := 10
				lr := float64(lrStep) * 1e-4
				wantMu := mu[0] - lr*(mu[0]+pe[0])
				wantLS := ls[0] - lr*(math.Exp(ls[0])-1.0)

				out := forwardActive(op, []int{1, lrStep}, mu, ls, pe)
				So(out, ShouldHaveLength, 2)
				So(out[0], ShouldAlmostEqual, wantMu, 1e-9)
				So(out[1], ShouldAlmostEqual, wantLS, 1e-9)
			})

			Convey("It should return a vector of length 2N", func() {
				n := 8
				mu := make([]float64, n)
				ls := make([]float64, n)
				pe := make([]float64, n)
				out := forwardActive(op, []int{n, 1}, mu, ls, pe)
				So(out, ShouldHaveLength, 2*n)
			})

			Convey("It should error when shape[1] yields non-positive lr", func() {
				mu := []float64{1.5, -0.5}
				ls := []float64{0.3, -0.3}
				pe := []float64{0.1, 0.2}
				stateDict := state.NewDict().
					WithShape([]int{2, 0}).
					WithInputs(mu, ls, pe)
				_, err := op.Forward(stateDict)

				So(err, ShouldNotBeNil)
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
				out := forwardActive(op, []int{3}, errVec, lp)
				So(out[0], ShouldAlmostEqual, 2.0, 1e-9)
				So(out[1], ShouldAlmostEqual, -3.0, 1e-9)
				So(out[2], ShouldAlmostEqual, 1.5, 1e-9)
			})

			Convey("It should amplify errors when precision is high", func() {
				errVec := []float64{1.0}
				lp := []float64{math.Log(10.0)} // precision = 10
				out := forwardActive(op, []int{1}, errVec, lp)
				So(out[0], ShouldAlmostEqual, 10.0, 1e-6)
			})

			Convey("It should attenuate errors when precision is low", func() {
				errVec := []float64{1.0}
				lp := []float64{math.Log(0.5)} // precision = 0.5
				out := forwardActive(op, []int{1}, errVec, lp)
				So(out[0], ShouldAlmostEqual, 0.5, 1e-6)
			})

			Convey("It should scale by very large exp(log_precision) without overflow for moderate lp", func() {
				errVec := []float64{1.0}
				lp := []float64{100.0}
				want := math.Exp(100.0)
				out := forwardActive(op, []int{1}, errVec, lp)
				// SIMD exp uses polynomial range reduction; ~1e-7 relative precision.
				relErr := math.Abs(out[0]-want) / want
				So(relErr, ShouldBeLessThan, 1e-6)
				So(math.IsInf(out[0], 0), ShouldBeFalse)
			})

			Convey("It should scale to near zero for very negative log_precision", func() {
				errVec := []float64{2.0}
				lp := []float64{-100.0}
				out := forwardActive(op, []int{1}, errVec, lp)
				So(out[0], ShouldAlmostEqual, 0.0, 1e-30)
			})

			Convey("It should error on mismatched error vs log_precision lengths", func() {
				stateDict := state.NewDict().
					WithShape([]int{2}).
					WithInputs([]float64{1, 2}, []float64{0.0})
				_, err := op.Forward(stateDict)

				So(err, ShouldNotBeNil)
			})

			Convey("It should return empty output when N is zero", func() {
				out := forwardActive(op, []int{0}, []float64{}, []float64{})
				So(out, ShouldHaveLength, 0)
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
				out := forwardActive(op, []int{1, 2}, q)
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
				out := forwardActive(op, []int{n, K}, q)
				expected := math.Log(float64(K)) / float64(K)
				So(out[0], ShouldAlmostEqual, expected, 1e-6)
			})

			Convey("It should return G of length K", func() {
				n, K := 3, 5
				q := make([]float64, n*K)
				for idx := range q {
					q[idx] = 1.0 / float64(K)
				}
				out := forwardActive(op, []int{n, K}, q)
				So(out, ShouldHaveLength, K)
			})

			Convey("It should stay finite for negative entries (clamped in kernel)", func() {
				q := []float64{-0.1, 1.1, 0.5, 0.5} // n=2, k=2 row-major
				out := forwardActive(op, []int{2, 2}, q)
				assertAllFiniteFloats(out)
			})

			Convey("It should stay finite when rows do not sum to one", func() {
				q := []float64{0.5, 0.5, 0.6, 0.6} // sums 1.2 per row
				out := forwardActive(op, []int{2, 2}, q)
				assertAllFiniteFloats(out)
			})

			Convey("It should stay finite with zero probabilities", func() {
				q := []float64{1.0, 0.0, 0.0, 1.0}
				out := forwardActive(op, []int{2, 2}, q)
				assertAllFiniteFloats(out)
			})

			Convey("It should stay finite for extremely small positive probabilities", func() {
				q := []float64{1.0 - 1e-300, 1e-300}
				out := forwardActive(op, []int{1, 2}, q)
				assertAllFiniteFloats(out)
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

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n}, mu, ls)
	}
}

func BenchmarkFreeEnergy_Forward_Small(b *testing.B) {
	op := NewFreeEnergy()
	n := 64
	mu := make([]float64, n)
	ls := make([]float64, n)

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n}, mu, ls)
	}
}

func BenchmarkFreeEnergy_Forward_Large(b *testing.B) {
	op := NewFreeEnergy()
	n := 1024
	mu := make([]float64, n)
	ls := make([]float64, n)

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n}, mu, ls)
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

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n, 1}, mu, ls, pe)
	}
}

func BenchmarkBeliefUpdate_Forward_Small(b *testing.B) {
	op := NewBeliefUpdate()
	n := 64
	mu := make([]float64, n)
	ls := make([]float64, n)
	pe := make([]float64, n)

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n, 1}, mu, ls, pe)
	}
}

func BenchmarkBeliefUpdate_Forward_Large(b *testing.B) {
	op := NewBeliefUpdate()
	n := 1024
	mu := make([]float64, n)
	ls := make([]float64, n)
	pe := make([]float64, n)

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n, 1}, mu, ls, pe)
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

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n}, errVec, lp)
	}
}

func BenchmarkPrecisionWeight_Forward_Small(b *testing.B) {
	op := NewPrecisionWeight()
	n := 64
	errVec := make([]float64, n)
	lp := make([]float64, n)

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n}, errVec, lp)
	}
}

func BenchmarkPrecisionWeight_Forward_Large(b *testing.B) {
	op := NewPrecisionWeight()
	n := 1024
	errVec := make([]float64, n)
	lp := make([]float64, n)

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n}, errVec, lp)
	}
}

func BenchmarkExpectedFreeEnergy_Forward(b *testing.B) {
	op := NewExpectedFreeEnergy()
	n, K := 32, 16
	q := make([]float64, n*K)

	for idx := range q {
		q[idx] = 1.0 / float64(K)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n, K}, q)
	}
}

func BenchmarkExpectedFreeEnergy_Forward_Small(b *testing.B) {
	op := NewExpectedFreeEnergy()
	n, K := 8, 4
	q := make([]float64, n*K)

	for idx := range q {
		q[idx] = 1.0 / float64(K)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n, K}, q)
	}
}

func BenchmarkExpectedFreeEnergy_Forward_Large(b *testing.B) {
	op := NewExpectedFreeEnergy()
	n, K := 64, 32
	q := make([]float64, n*K)

	for idx := range q {
		q[idx] = 1.0 / float64(K)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for idx := 0; idx < b.N; idx++ {
		forwardActive(op, []int{n, K}, q)
	}
}
