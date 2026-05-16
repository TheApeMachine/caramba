package adagrad

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestAdaGrad_Step(test *testing.T) {
	Convey("Given an AdaGrad optimizer", test, func() {
		Convey("Step", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				optimizer := NewAdaGrad()

				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads, accumulators := adagradParityState(parameterCount)
					initialAccumulators := append([]float64(nil), accumulators...)
					stateDict := adagradState(params, grads, 0.05, 1e-8, 0.01).
						WithV(accumulators)

					updated, err := optimizer.Step(stateDict)

					So(err, ShouldBeNil)

					for parameterIndex := range parameterCount {
						expectedParam, expectedAccumulator := adagradReferenceStep(
							params[parameterIndex],
							grads[parameterIndex],
							initialAccumulators[parameterIndex],
							0.05,
							1e-8,
							0.01,
						)
						So(updated.Out[parameterIndex], ShouldAlmostEqual, expectedParam, 1e-12)
						So(updated.V[parameterIndex], ShouldAlmostEqual, expectedAccumulator, 1e-12)
					}
				}
			})
		})
	})
}

func TestAdaDelta_Step(test *testing.T) {
	Convey("Given an AdaDelta optimizer", test, func() {
		Convey("Step", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					optimizer := NewAdaDelta(0.9, 1e-6, 0.01)
					params, grads, _ := adagradParityState(parameterCount)
					stateDict := adagradState(params, grads, 0, 0, 0)

					updated, err := optimizer.Step(stateDict)

					So(err, ShouldBeNil)

					for parameterIndex := range parameterCount {
						expectedParam, expectedEG2, expectedEDP2 := adadeltaReferenceStep(
							params[parameterIndex],
							grads[parameterIndex],
							0,
							0,
							0.9,
							1e-6,
							0.01,
						)
						So(updated.Out[parameterIndex], ShouldAlmostEqual, expectedParam, 1e-12)
						So(optimizer.eg2[parameterIndex], ShouldAlmostEqual, expectedEG2, 1e-12)
						So(optimizer.edp2[parameterIndex], ShouldAlmostEqual, expectedEDP2, 1e-12)
					}
				}
			})
		})
	})
}

func BenchmarkAdaGrad_Step(benchmark *testing.B) {
	optimizer := NewAdaGrad()
	parameterCount := 1 << 20
	params, grads, accumulators := adagradParityState(parameterCount)
	stateDict := adagradState(params, grads, 0.05, 1e-8, 0.01).
		WithV(accumulators)

	for benchmark.Loop() {
		updated, err := optimizer.Step(stateDict)

		if err != nil {
			benchmark.Fatalf("Step failed: %v", err)
		}

		stateDict.WithParams(updated.Out)
	}
}

func BenchmarkAdaDelta_Step(benchmark *testing.B) {
	optimizer := NewAdaDelta(0.9, 1e-6, 0.01)
	parameterCount := 1 << 20
	params, grads, _ := adagradParityState(parameterCount)
	stateDict := adagradState(params, grads, 0, 0, 0)

	for benchmark.Loop() {
		updated, err := optimizer.Step(stateDict)

		if err != nil {
			benchmark.Fatalf("Step failed: %v", err)
		}

		stateDict.WithParams(updated.Out)
	}
}

func adagradState(
	params []float64,
	grads []float64,
	learningRate float64,
	epsilon float64,
	weightDecay float64,
) *state.Dict {
	return state.NewDict().
		WithLR(learningRate).
		WithEps(epsilon).
		WithWD(weightDecay).
		WithParams(params).
		WithGrads(grads)
}

func adagradParityState(parameterCount int) ([]float64, []float64, []float64) {
	params := make([]float64, parameterCount)
	grads := make([]float64, parameterCount)
	accumulators := make([]float64, parameterCount)

	for parameterIndex := range parameterCount {
		params[parameterIndex] = float64(parameterIndex%17-8) * 0.125
		grads[parameterIndex] = float64(parameterIndex%11-5) * 0.0625
		accumulators[parameterIndex] = float64(parameterIndex%5) * 0.03125
	}

	return params, grads, accumulators
}

func adagradReferenceStep(
	param float64,
	grad float64,
	accumulator float64,
	learningRate float64,
	epsilon float64,
	weightDecay float64,
) (float64, float64) {
	effectiveGrad := grad + weightDecay*param
	updatedAccumulator := accumulator + effectiveGrad*effectiveGrad
	denominator := stdmath.Sqrt(updatedAccumulator) + epsilon
	updatedParam := param - learningRate*effectiveGrad/denominator

	return updatedParam, updatedAccumulator
}

func adadeltaReferenceStep(
	param float64,
	grad float64,
	eg2 float64,
	edp2 float64,
	rho float64,
	epsilon float64,
	weightDecay float64,
) (float64, float64, float64) {
	oneMinusRho := 1 - rho
	effectiveGrad := grad + weightDecay*param
	updatedEG2 := rho*eg2 + oneMinusRho*effectiveGrad*effectiveGrad
	delta := -stdmath.Sqrt(edp2+epsilon) / stdmath.Sqrt(updatedEG2+epsilon) * effectiveGrad
	updatedEDP2 := rho*edp2 + oneMinusRho*delta*delta
	updatedParam := param + delta

	return updatedParam, updatedEG2, updatedEDP2
}
