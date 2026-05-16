package lbfgs

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLBFGS_Step(test *testing.T) {
	Convey("Given an LBFGS optimizer", test, func() {
		Convey("Step", func() {
			Convey("It should match the first-step gradient descent path across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := lbfgsParityState(parameterCount)
					stateDict := state.NewDict().
						WithLR(0.03125).
						WithParams(params).
						WithGrads(grads)

					updated, err := NewLBFGS().Step(stateDict)

					So(err, ShouldBeNil)

					for parameterIndex := range parameterCount {
						expected := params[parameterIndex] - 0.03125*grads[parameterIndex]
						So(updated.Out[parameterIndex], ShouldAlmostEqual, expected, 1e-12)
					}
				}
			})
		})
	})
}

func TestLBFGS_Primitives(test *testing.T) {
	Convey("Given LBFGS vector primitives", test, func() {
		Convey("They should match scalar parity across SIMD lengths", func() {
			for _, elementCount := range []int{1, 7, 64, 1024, 8192} {
				left, right := lbfgsParityState(elementCount)

				sub := make([]float64, elementCount)
				lbfgsSub(sub, left, right)
				for elementIndex := range elementCount {
					So(sub[elementIndex], ShouldAlmostEqual, left[elementIndex]-right[elementIndex], 1e-12)
				}

				So(lbfgsDot(left, right), ShouldAlmostEqual, lbfgsReferenceDot(left, right), 1e-9)

				addScaled := append([]float64(nil), left...)
				lbfgsAddScaled(addScaled, right, -0.25)
				for elementIndex := range elementCount {
					So(addScaled[elementIndex], ShouldAlmostEqual, left[elementIndex]-0.25*right[elementIndex], 1e-12)
				}

				scaled := append([]float64(nil), left...)
				lbfgsScale(scaled, 0.5)
				for elementIndex := range elementCount {
					So(scaled[elementIndex], ShouldAlmostEqual, left[elementIndex]*0.5, 1e-12)
				}

				step := make([]float64, elementCount)
				lbfgsParamStep(step, left, right, 0.03125)
				for elementIndex := range elementCount {
					So(step[elementIndex], ShouldAlmostEqual, left[elementIndex]-0.03125*right[elementIndex], 1e-12)
				}
			}
		})
	})
}

func BenchmarkLBFGS_Step(benchmark *testing.B) {
	params, grads := lbfgsParityState(1 << 20)
	stateDict := state.NewDict().
		WithLR(0.03125).
		WithParams(params).
		WithGrads(grads)
	optimizer := NewLBFGS()

	for benchmark.Loop() {
		updated, err := optimizer.Step(stateDict)

		if err != nil {
			benchmark.Fatalf("Step failed: %v", err)
		}

		stateDict.WithParams(updated.Out)
	}
}

func lbfgsParityState(parameterCount int) ([]float64, []float64) {
	params := make([]float64, parameterCount)
	grads := make([]float64, parameterCount)

	for parameterIndex := range parameterCount {
		params[parameterIndex] = float64(parameterIndex%17-8) * 0.125
		grads[parameterIndex] = float64(parameterIndex%11-5) * 0.0625
	}

	return params, grads
}

func lbfgsReferenceDot(left []float64, right []float64) float64 {
	sum := 0.0

	for elementIndex := range left {
		sum += left[elementIndex] * right[elementIndex]
	}

	return sum
}
