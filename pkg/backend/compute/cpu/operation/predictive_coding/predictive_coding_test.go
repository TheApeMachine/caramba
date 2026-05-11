package predictive_coding_test

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	pc "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/predictive_coding"
)

func TestPrediction(t *testing.T) {
	Convey("Given a Prediction operation", t, func() {
		dOut, dIn := 4, 8
		W := make([]float64, dOut*dIn)
		r := make([]float64, dIn)

		// Identity-like block: W[i, i] = 1 for i < dOut
		for i := 0; i < dOut; i++ {
			W[i*dIn+i] = 1.0
		}
		for i := range r {
			r[i] = float64(i + 1)
		}

		Convey("It should compute W @ r correctly", func() {
			op := pc.NewPrediction()
			out := op.Forward([]int{dOut, dIn}, W, r)

			So(len(out), ShouldEqual, dOut)

			for i := 0; i < dOut; i++ {
				So(out[i], ShouldAlmostEqual, float64(i+1), 1e-9)
			}
		})

		Convey("It should panic on shape mismatch", func() {
			op := pc.NewPrediction()
			So(func() { op.Forward([]int{dOut, dIn}, W, r[:dIn-1]) }, ShouldPanic)
		})
	})
}

func TestPredictionError(t *testing.T) {
	Convey("Given a PredictionError operation", t, func() {
		n := 16
		x := make([]float64, n)
		muHat := make([]float64, n)
		prec := make([]float64, n)

		for i := range x {
			x[i] = float64(i) * 0.1
			muHat[i] = float64(i) * 0.09
			prec[i] = 2.0
		}

		Convey("It should compute unweighted prediction error", func() {
			op := pc.NewPredictionError()
			out := op.Forward([]int{n}, x, muHat)

			So(len(out), ShouldEqual, n)

			for i := range out {
				So(out[i], ShouldAlmostEqual, x[i]-muHat[i], 1e-9)
			}
		})

		Convey("It should compute precision-weighted prediction error", func() {
			op := pc.NewPredictionError()
			out := op.Forward([]int{n}, x, muHat, prec)

			So(len(out), ShouldEqual, n)

			for i := range out {
				expected := prec[i] * (x[i] - muHat[i])
				So(out[i], ShouldAlmostEqual, expected, 1e-9)
			}
		})
	})
}

func TestUpdateRepresentation(t *testing.T) {
	Convey("Given an UpdateRepresentation operation", t, func() {
		dIn, dOut := 4, 8
		W := make([]float64, dOut*dIn)
		r := make([]float64, dIn)
		epsLower := make([]float64, dOut)
		epsSelf := make([]float64, dIn)
		lr := []float64{0.1}

		for i := range r {
			r[i] = 1.0
		}

		// W = zero: W^T @ eps_lower = 0; eps_self = 0; r_new = r + 0 = r
		Convey("It should leave r unchanged when all errors are zero", func() {
			op := pc.NewUpdateRepresentation()
			out := op.Forward([]int{dIn, dOut}, r, W, epsLower, epsSelf, lr)

			So(len(out), ShouldEqual, dIn)

			for i, v := range out {
				So(v, ShouldAlmostEqual, r[i], 1e-9)
			}
		})

		Convey("It should update r when eps_self is non-zero", func() {
			for i := range epsSelf {
				epsSelf[i] = 0.5
			}

			op := pc.NewUpdateRepresentation()
			out := op.Forward([]int{dIn, dOut}, r, W, epsLower, epsSelf, lr)

			So(len(out), ShouldEqual, dIn)

			// r_new[i] = 1.0 + 0.1 * (0 - 0.5) = 0.95
			for _, v := range out {
				So(v, ShouldAlmostEqual, 0.95, 1e-9)
			}
		})
	})
}

func TestUpdateWeights(t *testing.T) {
	Convey("Given an UpdateWeights operation", t, func() {
		dOut, dIn := 3, 4
		W := make([]float64, dOut*dIn)
		eps := make([]float64, dOut)
		r := make([]float64, dIn)
		lr := []float64{1.0}

		eps[0] = 1.0
		r[0] = 1.0

		Convey("It should update W by the outer product of eps and r", func() {
			op := pc.NewUpdateWeights()
			out := op.Forward([]int{dOut, dIn}, W, eps, r, lr)

			So(len(out), ShouldEqual, dOut*dIn)
			// W_new[0,0] = 0 + 1.0 * 1.0 * 1.0 = 1.0
			So(out[0], ShouldAlmostEqual, 1.0, 1e-9)
			// All other entries zero since eps[1..] = 0 and r[1..] = 0
			for i := 1; i < len(out); i++ {
				So(out[i], ShouldAlmostEqual, 0.0, 1e-9)
			}
		})

		Convey("It should converge prediction error toward zero over multiple steps", func() {
			// Simple 1D test: x = 1.0, mu_hat from W@r should converge
			opPred := pc.NewPrediction()
			opErr := pc.NewPredictionError()
			opWUpdate := pc.NewUpdateWeights()

			W1 := []float64{0.5}
			r1 := []float64{1.0}
			x1 := []float64{1.0}

			for step := 0; step < 200; step++ {
				muHat := opPred.Forward([]int{1, 1}, W1, r1)
				eps := opErr.Forward([]int{1}, x1, muHat)
				W1 = opWUpdate.Forward([]int{1, 1}, W1, eps, r1, []float64{0.1})
			}

			// After convergence W should be ~1.0 since r=1, x=1
			So(math.Abs(W1[0]-1.0), ShouldBeLessThan, 0.01)
		})
	})
}

func BenchmarkPrediction(b *testing.B) {
	dOut, dIn := 256, 1024
	W := make([]float64, dOut*dIn)
	r := make([]float64, dIn)

	for i := range W {
		W[i] = float64(i) / float64(len(W))
	}

	for i := range r {
		r[i] = float64(i) / float64(len(r))
	}

	op := pc.NewPrediction()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward([]int{dOut, dIn}, W, r)
	}
}

func BenchmarkPredictionError(b *testing.B) {
	n := 4096
	x := make([]float64, n)
	muHat := make([]float64, n)
	prec := make([]float64, n)

	for i := range x {
		x[i] = float64(i) * 0.001
		muHat[i] = float64(i) * 0.0009
		prec[i] = 1.0
	}

	op := pc.NewPredictionError()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward([]int{n}, x, muHat, prec)
	}
}

func BenchmarkUpdateWeights(b *testing.B) {
	dOut, dIn := 256, 1024
	W := make([]float64, dOut*dIn)
	eps := make([]float64, dOut)
	r := make([]float64, dIn)
	lr := []float64{0.01}

	for i := range eps {
		eps[i] = float64(i) * 0.001
	}

	for i := range r {
		r[i] = float64(i) * 0.001
	}

	op := pc.NewUpdateWeights()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward([]int{dOut, dIn}, W, eps, r, lr)
	}
}
