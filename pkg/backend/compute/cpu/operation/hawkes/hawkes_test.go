package hawkes

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestIntensity(t *testing.T) {
	Convey("Given an Intensity operation", t, func() {
		op := NewIntensity()

		Convey("Forward", func() {
			Convey("It should compute baseline intensity with no history", func() {
				// K=1, T=0 (empty history): lambda = mu
				shape := []int{1, 0}
				times := []float64{}
				alpha := []float64{0.5}
				beta := []float64{1.0}
				mu := []float64{2.0}
				tVal := []float64{5.0}
				out := op.Forward(shape, times, alpha, beta, mu, tVal)
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldAlmostEqual, 2.0, 1e-9)
			})

			Convey("It should include excitation from past events", func() {
				// K=1, single event at t=0, query at t=1
				// lambda(1) = mu + alpha * exp(-beta * 1)
				shape := []int{1, 1}
				times := []float64{0.0}
				alpha := []float64{1.0}
				beta := []float64{1.0}
				mu := []float64{0.5}
				tVal := []float64{1.0}
				out := op.Forward(shape, times, alpha, beta, mu, tVal)
				expected := 0.5 + stdmath.Exp(-1.0)
				So(out[0], ShouldAlmostEqual, expected, 1e-9)
			})

			Convey("It should handle K > 1 processes independently", func() {
				shape := []int{2, 1}
				times := []float64{0.0}
				alpha := []float64{1.0, 2.0}
				beta := []float64{1.0, 0.5}
				mu := []float64{0.1, 0.2}
				tVal := []float64{2.0}
				out := op.Forward(shape, times, alpha, beta, mu, tVal)
				So(out, ShouldHaveLength, 2)
				e0 := 0.1 + 1.0*stdmath.Exp(-1.0*2.0)
				e1 := 0.2 + 2.0*stdmath.Exp(-0.5*2.0)
				So(out[0], ShouldAlmostEqual, e0, 1e-9)
				So(out[1], ShouldAlmostEqual, e1, 1e-9)
			})

			Convey("It should panic on insufficient data", func() {
				So(func() { op.Forward([]int{1, 0}, []float64{}) }, ShouldPanic)
			})
		})
	})
}

func TestKernelMatrix(t *testing.T) {
	Convey("Given a KernelMatrix operation", t, func() {
		op := NewKernelMatrix()

		Convey("Forward", func() {
			Convey("It should produce zero diagonal and lower triangle", func() {
				times := []float64{0.0, 1.0, 2.0}
				alpha := []float64{1.0}
				beta := []float64{0.5}
				shape := []int{3, 0, 0}
				out := op.Forward(shape, times, alpha, beta)
				So(out, ShouldHaveLength, 9)
				// Diagonal and below must be zero
				So(out[0], ShouldEqual, 0) // [0,0]
				So(out[3], ShouldEqual, 0) // [1,0]
				So(out[4], ShouldEqual, 0) // [1,1]
				So(out[6], ShouldEqual, 0) // [2,0]
				So(out[7], ShouldEqual, 0) // [2,1]
				So(out[8], ShouldEqual, 0) // [2,2]
			})

			Convey("It should compute correct upper triangle values", func() {
				times := []float64{0.0, 1.0}
				alpha := []float64{2.0}
				beta := []float64{1.0}
				shape := []int{2, 0, 0}
				out := op.Forward(shape, times, alpha, beta)
				// K[0,1] = 2 * exp(-1.0 * (1-0)) = 2*exp(-1)
				expected := 2.0 * stdmath.Exp(-1.0)
				So(out[1], ShouldAlmostEqual, expected, 1e-9)
			})
		})
	})
}

func TestLogLikelihood(t *testing.T) {
	Convey("Given a LogLikelihood operation", t, func() {
		op := NewLogLikelihood()

		Convey("Forward", func() {
			Convey("It should compute sum log(lambda) - integral", func() {
				T := 3
				times := []float64{1.0, 2.0, 3.0}
				intensities := []float64{1.0, 2.0, 4.0}
				integral := []float64{5.0}
				expected := stdmath.Log(1.0) + stdmath.Log(2.0) + stdmath.Log(4.0) - 5.0
				out := op.Forward([]int{T, 0}, times, intensities, integral)
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldAlmostEqual, expected, 1e-9)
			})

			Convey("It should panic on data length mismatch", func() {
				So(func() {
					op.Forward([]int{3, 0},
						[]float64{1, 2, 3},
						[]float64{1, 2},
						[]float64{0})
				}, ShouldPanic)
			})
		})
	})
}

func TestSimulate(t *testing.T) {
	Convey("Given a Simulate operation", t, func() {
		op := NewSimulate()

		Convey("Forward", func() {
			Convey("It should generate events within [0, T_max]", func() {
				K, maxSteps := 1, 200
				mu := []float64{1.0}
				alpha := []float64{0.5}
				beta := []float64{2.0}
				tMax := []float64{10.0}
				shape := []int{K, maxSteps}
				out := op.Forward(shape, mu, alpha, beta, tMax)
				So(out, ShouldHaveLength, maxSteps)

				for _, ev := range out {
					if ev == -1 {
						break
					}

					So(ev, ShouldBeLessThan, 10.0)
					So(ev, ShouldBeGreaterThanOrEqualTo, 0.0)
				}
			})

			Convey("It should produce events in ascending order", func() {
				K, maxSteps := 1, 100
				mu := []float64{2.0}
				alpha := []float64{0.3}
				beta := []float64{1.0}
				tMax := []float64{5.0}
				out := op.Forward([]int{K, maxSteps}, mu, alpha, beta, tMax)

				prev := -1.0

				for _, ev := range out {
					if ev == -1 {
						break
					}

					So(ev, ShouldBeGreaterThan, prev)
					prev = ev
				}
			})

			Convey("It should panic on insufficient data", func() {
				So(func() { op.Forward([]int{1, 10}) }, ShouldPanic)
			})
		})
	})
}

func BenchmarkIntensity_Forward(b *testing.B) {
	op := NewIntensity()
	T := 1000
	K := 4
	times := make([]float64, T)
	alpha := make([]float64, K)
	beta := make([]float64, K)
	mu := make([]float64, K)

	for idx := range T {
		times[idx] = float64(idx) * 0.01
	}

	for k := range K {
		alpha[k] = 0.5
		beta[k] = 1.0
		mu[k] = 0.5
	}

	tVal := []float64{float64(T) * 0.01}
	shape := []int{K, T}
	b.ResetTimer()

	for range b.N {
		op.Forward(shape, times, alpha, beta, mu, tVal)
	}
}

func BenchmarkKernelMatrix_Forward(b *testing.B) {
	op := NewKernelMatrix()
	T := 256
	times := make([]float64, T)

	for idx := range T {
		times[idx] = float64(idx) * 0.05
	}

	alpha := []float64{1.0}
	beta := []float64{0.5}
	shape := []int{T, 0, 0}
	b.ResetTimer()

	for range b.N {
		op.Forward(shape, times, alpha, beta)
	}
}

func BenchmarkSimulate_Forward(b *testing.B) {
	op := NewSimulate()
	K, maxSteps := 4, 500
	mu := []float64{1.0, 0.8, 1.2, 0.9}
	alpha := []float64{0.4, 0.3, 0.5, 0.6}
	beta := []float64{1.0, 1.5, 0.8, 1.2}
	tMax := []float64{20.0}
	shape := []int{K, maxSteps}
	b.ResetTimer()

	for range b.N {
		op.Forward(shape, mu, alpha, beta, tMax)
	}
}
