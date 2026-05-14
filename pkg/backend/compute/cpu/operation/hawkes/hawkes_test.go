package hawkes

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type stateOperation interface {
	Forward(*state.Dict) (*state.Dict, error)
}

func forwardHawkes(
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

func forwardHawkesErr(
	operation stateOperation, shape []int, inputs ...[]float64,
) error {
	stateDict := state.NewDict().WithShape(shape)
	values := make([]any, len(inputs))

	for index := range inputs {
		values[index] = inputs[index]
	}

	stateDict.WithInputs(values...)
	_, err := operation.Forward(stateDict)

	return err
}

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
				out := forwardHawkes(op, shape, times, alpha, beta, mu, tVal)
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
				out := forwardHawkes(op, shape, times, alpha, beta, mu, tVal)
				expected := 0.5 + math.Exp(-1.0)
				So(out[0], ShouldAlmostEqual, expected, 1e-9)
			})

			Convey("It should handle K > 1 processes independently", func() {
				shape := []int{2, 1}
				times := []float64{0.0}
				alpha := []float64{1.0, 2.0}
				beta := []float64{1.0, 0.5}
				mu := []float64{0.1, 0.2}
				tVal := []float64{2.0}
				out := forwardHawkes(op, shape, times, alpha, beta, mu, tVal)
				So(out, ShouldHaveLength, 2)
				e0 := 0.1 + 1.0*math.Exp(-1.0*2.0)
				e1 := 0.2 + 2.0*math.Exp(-0.5*2.0)
				So(out[0], ShouldAlmostEqual, e0, 1e-9)
				So(out[1], ShouldAlmostEqual, e1, 1e-9)
			})

			Convey("It should error on insufficient data", func() {
				err := forwardHawkesErr(op, []int{1, 0}, []float64{})

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "input")
			})
		})
	})
}

func TestApplyIntensity(t *testing.T) {
	Convey("Given the architecture-specific Hawkes intensity path", t, func() {
		times := []float64{0.0, 0.2, 0.4, 0.8, 1.0}
		alpha := []float64{0.5, 1.2}
		beta := []float64{1.0, 0.7}
		mu := []float64{0.1, 0.3}
		actual := make([]float64, len(alpha))
		expected := make([]float64, len(alpha))

		Convey("It should match the scalar reference within 1e-12", func() {
			applyIntensity(actual, times, alpha, beta, mu, 1.1, len(alpha), len(times))
			applyIntensityScalar(expected, times, alpha, beta, mu, 1.1, len(alpha), len(times))

			for index := range actual {
				So(actual[index], ShouldAlmostEqual, expected[index], 1e-12)
			}
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
				out := forwardHawkes(op, shape, times, alpha, beta)
				So(out, ShouldHaveLength, 9)
				// Diagonal and below must be zero
				So(out[0], ShouldEqual, 0) // [0,0]
				So(out[3], ShouldEqual, 0) // [1,0]
				So(out[4], ShouldEqual, 0) // [1,1]
				So(out[6], ShouldEqual, 0) // [2,0]
				So(out[7], ShouldEqual, 0) // [2,1]
				So(out[8], ShouldEqual, 0) // [2,2]
				// Upper triangle [0,2] and [1,2]
				So(out[2], ShouldBeGreaterThan, 0)
				So(out[5], ShouldBeGreaterThan, 0)
				So(out[2], ShouldAlmostEqual, math.Exp(-0.5*(2.0-0.0)), 1e-12)
				So(out[5], ShouldAlmostEqual, math.Exp(-0.5*(2.0-1.0)), 1e-12)
			})

			Convey("It should compute correct upper triangle values", func() {
				times := []float64{0.0, 1.0}
				alpha := []float64{2.0}
				beta := []float64{1.0}
				shape := []int{2, 0, 0}
				out := forwardHawkes(op, shape, times, alpha, beta)
				// K[0,1] = 2 * exp(-1.0 * (1-0)) = 2*exp(-1)
				expected := 2.0 * math.Exp(-1.0)
				So(out[1], ShouldAlmostEqual, expected, 1e-9)
			})
		})
	})
}

func TestApplyKernelMatrix(t *testing.T) {
	Convey("Given the architecture-specific Hawkes kernel matrix path", t, func() {
		times := []float64{0.0, 0.25, 0.75, 1.5, 2.25}
		actual := make([]float64, len(times)*len(times))
		expected := make([]float64, len(times)*len(times))

		Convey("It should match the scalar reference within 1e-12", func() {
			applyKernelMatrix(actual, times, 0.8, 0.4, len(times))
			applyKernelMatrixScalar(expected, times, 0.8, 0.4, len(times))

			for index := range actual {
				So(actual[index], ShouldAlmostEqual, expected[index], 1e-12)
			}
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
				expected := math.Log(1.0) + math.Log(2.0) + math.Log(4.0) - 5.0
				out := forwardHawkes(op, []int{T}, times, intensities, integral)
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldAlmostEqual, expected, 1e-9)
			})

			Convey("It should error on data length mismatch", func() {
				err := forwardHawkesErr(op, []int{3},
					[]float64{1, 2, 3},
					[]float64{1, 2},
					[]float64{0})

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "len(intensities)=2, need T=3")
			})
		})
	})
}

func TestSimulate(t *testing.T) {
	const sentinelUnusedEvent = -1.0

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
				out := forwardHawkes(op, shape, mu, alpha, beta, tMax)
				So(out, ShouldHaveLength, maxSteps)

				for _, ev := range out {
					if ev == sentinelUnusedEvent {
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
				out := forwardHawkes(op, []int{K, maxSteps}, mu, alpha, beta, tMax)

				prev := -1.0

				for _, ev := range out {
					if ev == sentinelUnusedEvent {
						break
					}

					So(ev, ShouldBeGreaterThan, prev)
					prev = ev
				}
			})

			Convey("It should error on insufficient data", func() {
				err := forwardHawkesErr(op, []int{1, 10})

				So(err, ShouldNotBeNil)
			})
		})
	})
}

func BenchmarkIntensity_Forward(b *testing.B) {
	op := NewIntensity()
	// T=1000 and K=4: representative multi-process intensity at fixed wall time with dense history.
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
	b.ReportAllocs()

	for b.Loop() {
		forwardHawkes(op, shape, times, alpha, beta, mu, tVal)
	}
}

func BenchmarkKernelMatrix_Forward(b *testing.B) {
	op := NewKernelMatrix()
	// T=256: moderate stress for O(T²) upper-triangle fill without dominating benchmark time.
	T := 256
	times := make([]float64, T)

	for idx := range T {
		times[idx] = float64(idx) * 0.05
	}

	alpha := []float64{1.0}
	beta := []float64{0.5}
	shape := []int{T, 0, 0}
	b.ResetTimer()
	b.ReportAllocs()

	for b.Loop() {
		forwardHawkes(op, shape, times, alpha, beta)
	}
}

func BenchmarkLogLikelihood_Forward(b *testing.B) {
	op := NewLogLikelihood()
	T := 1000
	times := make([]float64, T)
	intensities := make([]float64, T)

	for idx := range T {
		times[idx] = float64(idx) * 0.01
		intensities[idx] = 0.5 + float64(idx)*1e-4
	}

	integral := []float64{12.0}
	shape := []int{T}
	b.ResetTimer()
	b.ReportAllocs()

	for b.Loop() {
		forwardHawkes(op, shape, times, intensities, integral)
	}
}

func BenchmarkSimulate_Forward(b *testing.B) {
	op := NewSimulate()
	// K=4 and maxSteps=500: multi-series thinning with bounded event buffer (realistic iteration cost).
	K, maxSteps := 4, 500
	mu := []float64{1.0, 0.8, 1.2, 0.9}
	alpha := []float64{0.4, 0.3, 0.5, 0.6}
	beta := []float64{1.0, 1.5, 0.8, 1.2}
	tMax := []float64{20.0}
	shape := []int{K, maxSteps}
	b.ResetTimer()
	b.ReportAllocs()

	for b.Loop() {
		forwardHawkes(op, shape, mu, alpha, beta, tMax)
	}
}
