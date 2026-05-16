package lars

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLARS_Step(test *testing.T) {
	Convey("Given a LARS optimizer", test, func() {
		Convey("Step", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := larsParityState(parameterCount)
					stateDict := larsState(params, grads)

					updated, err := NewLARS().Step(stateDict)

					So(err, ShouldBeNil)

					localLR := larsReferenceLocalLR(params, grads, 0.001, 0.01, 1e-6)

					for parameterIndex := range parameterCount {
						velocity := localLR * (grads[parameterIndex] + 0.01*params[parameterIndex])
						So(updated.M[parameterIndex], ShouldAlmostEqual, velocity, 1e-12)
						So(updated.Out[parameterIndex], ShouldAlmostEqual, params[parameterIndex]-velocity, 1e-12)
					}
				}
			})
		})
	})
}

func TestLAMB_Step(test *testing.T) {
	Convey("Given a LAMB optimizer", test, func() {
		Convey("Step", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := larsParityState(parameterCount)
					stateDict := state.NewDict().
						WithParams(params).
						WithGrads(grads)
					optimizer := NewLAMB(0.01, 0.9, 0.999, 1e-8, 0.01)

					updated, err := optimizer.Step(stateDict)

					So(err, ShouldBeNil)

					expectedM, expectedV, expectedOut := lambReferenceFirstStep(params, grads)

					for parameterIndex := range parameterCount {
						So(optimizer.m[parameterIndex], ShouldAlmostEqual, expectedM[parameterIndex], 1e-12)
						So(optimizer.v[parameterIndex], ShouldAlmostEqual, expectedV[parameterIndex], 1e-12)
						So(updated.Out[parameterIndex], ShouldAlmostEqual, expectedOut[parameterIndex], 1e-12)
					}
				}
			})
		})
	})
}

func BenchmarkLARS_Step(benchmark *testing.B) {
	params, grads := larsParityState(1 << 20)
	stateDict := larsState(params, grads)
	optimizer := NewLARS()

	for benchmark.Loop() {
		updated, err := optimizer.Step(stateDict)

		if err != nil {
			benchmark.Fatalf("Step failed: %v", err)
		}

		stateDict.WithParams(updated.Out)
	}
}

func BenchmarkLAMB_Step(benchmark *testing.B) {
	params, grads := larsParityState(1 << 20)
	stateDict := state.NewDict().
		WithParams(params).
		WithGrads(grads)
	optimizer := NewLAMB(0.01, 0.9, 0.999, 1e-8, 0.01)

	for benchmark.Loop() {
		updated, err := optimizer.Step(stateDict)

		if err != nil {
			benchmark.Fatalf("Step failed: %v", err)
		}

		stateDict.WithParams(updated.Out)
	}
}

func larsState(params []float64, grads []float64) *state.Dict {
	return state.NewDict().
		WithLR(0.01).
		WithEta(0.001).
		WithEps(1e-6).
		WithMomentum(0.9).
		WithWD(0.01).
		WithParams(params).
		WithGrads(grads)
}

func larsParityState(parameterCount int) ([]float64, []float64) {
	params := make([]float64, parameterCount)
	grads := make([]float64, parameterCount)

	for parameterIndex := range parameterCount {
		params[parameterIndex] = float64(parameterIndex%17-8) * 0.125
		grads[parameterIndex] = float64(parameterIndex%11-5) * 0.0625
	}

	return params, grads
}

func larsReferenceLocalLR(
	params []float64,
	grads []float64,
	eta float64,
	weightDecay float64,
	epsilon float64,
) float64 {
	paramNorm := stdmath.Sqrt(larsReferenceNormSq(params))
	gradNorm := stdmath.Sqrt(larsReferenceNormSq(grads))

	if paramNorm > 0 && gradNorm > 0 {
		return eta * paramNorm / (gradNorm + weightDecay*paramNorm + epsilon)
	}

	return 0.01
}

func lambReferenceFirstStep(params []float64, grads []float64) ([]float64, []float64, []float64) {
	m := make([]float64, len(params))
	v := make([]float64, len(params))
	update := make([]float64, len(params))
	out := make([]float64, len(params))

	for parameterIndex := range params {
		m[parameterIndex] = 0.1 * grads[parameterIndex]
		v[parameterIndex] = 0.001 * grads[parameterIndex] * grads[parameterIndex]
		mHat := m[parameterIndex] * 10
		vHat := v[parameterIndex] * 1000
		update[parameterIndex] = mHat/(stdmath.Sqrt(vHat)+1e-8) + 0.01*params[parameterIndex]
	}

	paramNorm := stdmath.Sqrt(larsReferenceNormSq(params))
	updateNorm := stdmath.Sqrt(larsReferenceNormSq(update))
	ratio := 0.01

	if paramNorm > 0 && updateNorm > 0 {
		ratio = 0.01 * paramNorm / updateNorm
	}

	for parameterIndex := range params {
		out[parameterIndex] = params[parameterIndex] - ratio*update[parameterIndex]
	}

	return m, v, out
}

func larsReferenceNormSq(values []float64) float64 {
	sum := 0.0

	for _, value := range values {
		sum += value * value
	}

	return sum
}
