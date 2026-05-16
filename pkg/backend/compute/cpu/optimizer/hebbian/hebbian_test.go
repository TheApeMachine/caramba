package hebbian

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestHebbian_Step(test *testing.T) {
	Convey("Given a Hebbian optimizer", test, func() {
		Convey("Step", func() {
			Convey("It should strengthen correlated weights", func() {
				optimizer := NewHebbian()
				stateDict := hebbianState([]float64{0.0}, []float64{1.0}, 0.1, 0)

				updated, err := optimizer.Step(stateDict)

				So(err, ShouldBeNil)
				So(updated.Out[0], ShouldAlmostEqual, 0.1)
			})

			Convey("It should clip weights when MaxNorm is set", func() {
				optimizer := NewHebbian()
				stateDict := hebbianState(
					[]float64{0.5, 0.5, 0.5, 0.5},
					[]float64{1.0, 1.0, 1.0, 1.0},
					1.0,
					1.0,
				)

				updated, err := optimizer.Step(stateDict)

				So(err, ShouldBeNil)
				norm := 0.0
				for _, value := range updated.Out {
					norm += value * value
				}
				So(stdmath.Sqrt(norm), ShouldBeLessThanOrEqualTo, 1.0+1e-9)
			})

			Convey("It should match scalar parity across SIMD lengths", func() {
				optimizer := NewHebbian()

				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := hebbianParityState(parameterCount)
					stateDict := hebbianState(params, grads, 0.03125, 0)

					updated, err := optimizer.Step(stateDict)

					So(err, ShouldBeNil)

					for parameterIndex := range parameterCount {
						expectedParam := hebbianReferenceStep(
							params[parameterIndex],
							grads[parameterIndex],
							0.03125,
						)
						So(updated.Out[parameterIndex], ShouldAlmostEqual, expectedParam, 1e-12)
					}
				}
			})

			Convey("It should match scalar norm clipping across SIMD lengths", func() {
				optimizer := NewHebbian()

				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := hebbianParityState(parameterCount)
					stateDict := hebbianState(params, grads, 0.0625, 3.0)

					updated, err := optimizer.Step(stateDict)

					So(err, ShouldBeNil)

					expectedOut := hebbianReferenceClipped(params, grads, 0.0625, 3.0)

					for parameterIndex := range parameterCount {
						So(updated.Out[parameterIndex], ShouldAlmostEqual, expectedOut[parameterIndex], 1e-12)
					}
				}
			})
		})
	})
}

func TestOjaRule_Step(test *testing.T) {
	Convey("Given an Oja rule optimizer", test, func() {
		Convey("Step", func() {
			Convey("It should keep weights bounded via decay", func() {
				optimizer := NewOjaRule(0.01)
				params := make([]float64, 4)
				for parameterIndex := range params {
					params[parameterIndex] = 0.5
				}
				for range 5000 {
					// simulate unit post-synaptic activity
					grads := make([]float64, 4)
					for parameterIndex := range grads {
						grads[parameterIndex] = params[parameterIndex] // post*pre ≈ p
					}
					params = optimizer.Step(params, grads)
				}
				// weight norm should converge to ~1.0 (unit sphere)
				norm := 0.0
				for _, value := range params {
					norm += value * value
				}
				So(stdmath.Sqrt(norm), ShouldAlmostEqual, 1.0, 0.1)
			})

			Convey("It should match scalar parity across SIMD lengths", func() {
				optimizer := NewOjaRule(0.03125)

				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := hebbianParityState(parameterCount)
					updated := optimizer.Step(params, grads)
					postSq := hebbianReferenceSumSq(grads)

					for parameterIndex := range parameterCount {
						expectedParam := hebbianReferenceOja(
							params[parameterIndex],
							grads[parameterIndex],
							0.03125,
							postSq,
						)
						So(updated[parameterIndex], ShouldAlmostEqual, expectedParam, 1e-12)
					}
				}
			})
		})
	})
}

func TestBCM_Step(test *testing.T) {
	Convey("Given a BCM optimizer", test, func() {
		Convey("Step", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					optimizer := NewBCM(0.03125, 8.0)
					params, grads := hebbianParityState(parameterCount)
					updated := optimizer.Step(params, grads)
					postSq := hebbianReferenceSumSq(grads)
					theta := postSq / 8.0
					factor := 0.03125 * (postSq - theta)

					for parameterIndex := range parameterCount {
						expectedParam := hebbianReferenceStep(
							params[parameterIndex],
							grads[parameterIndex],
							factor,
						)
						So(updated[parameterIndex], ShouldAlmostEqual, expectedParam, 1e-12)
					}
				}
			})
		})
	})
}

func BenchmarkHebbian_Step(benchmark *testing.B) {
	optimizer := NewHebbian()
	parameterCount := 1 << 20
	params := make([]float64, parameterCount)
	grads := make([]float64, parameterCount)
	for parameterIndex := range params {
		params[parameterIndex] = float64(parameterIndex) * 1e-6
		grads[parameterIndex] = float64(parameterIndex%2*2-1) * 1e-4
	}
	stateDict := hebbianState(params, grads, 0.01, 1.0)
	benchmark.ResetTimer()
	for benchmark.Loop() {
		updated, err := optimizer.Step(stateDict)
		if err != nil {
			benchmark.Fatalf("Step failed: %v", err)
		}
		stateDict.WithParams(updated.Out)
	}
}

func BenchmarkOjaRule_Step(benchmark *testing.B) {
	optimizer := NewOjaRule(0.01)
	parameterCount := 1 << 20
	params, grads := hebbianParityState(parameterCount)

	for benchmark.Loop() {
		params = optimizer.Step(params, grads)
	}
}

func BenchmarkBCM_Step(benchmark *testing.B) {
	optimizer := NewBCM(0.01, 16.0)
	parameterCount := 1 << 20
	params, grads := hebbianParityState(parameterCount)

	for benchmark.Loop() {
		params = optimizer.Step(params, grads)
	}
}

func hebbianState(params, grads []float64, lr, maxNorm float64) *state.Dict {
	return state.NewDict().
		WithLR(lr).
		WithMaxNorm(maxNorm).
		WithParams(params).
		WithGrads(grads)
}

func hebbianParityState(parameterCount int) ([]float64, []float64) {
	params := make([]float64, parameterCount)
	grads := make([]float64, parameterCount)

	for parameterIndex := range parameterCount {
		params[parameterIndex] = float64(parameterIndex%17-8) * 0.125
		grads[parameterIndex] = float64(parameterIndex%11-5) * 0.0625
	}

	return params, grads
}

func hebbianReferenceStep(param float64, grad float64, learningRate float64) float64 {
	return param + learningRate*grad
}

func hebbianReferenceClipped(
	params []float64,
	grads []float64,
	learningRate float64,
	maxNorm float64,
) []float64 {
	out := make([]float64, len(params))
	normSq := 0.0

	for parameterIndex := range params {
		out[parameterIndex] = hebbianReferenceStep(
			params[parameterIndex],
			grads[parameterIndex],
			learningRate,
		)
		normSq += out[parameterIndex] * out[parameterIndex]
	}

	norm := stdmath.Sqrt(normSq)

	if norm <= maxNorm {
		return out
	}

	scale := maxNorm / norm

	for parameterIndex := range out {
		out[parameterIndex] *= scale
	}

	return out
}

func hebbianReferenceOja(
	param float64,
	grad float64,
	learningRate float64,
	postSq float64,
) float64 {
	return param + learningRate*grad - learningRate*postSq*param
}

func hebbianReferenceSumSq(values []float64) float64 {
	sumSq := 0.0

	for _, value := range values {
		sumSq += value * value
	}

	return sumSq
}
