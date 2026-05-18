package adam

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestAdaMax_Step(t *testing.T) {
	Convey("Given an AdaMax optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should update params and infinity-norm state on the first step", func() {
				params := []float64{1.0, -2.0, 0.5}
				grads := []float64{0.1, -0.2, 0.05}
				stateDict := adaMaxState(params, grads)
				opt := NewAdaMax()

				updated, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				So(stateDict.Step, ShouldEqual, 1)
				assertFloat64Values(updated.Out, adaMaxExpectedParams(params, grads))
				assertFloat64Values(stateDict.Out, adaMaxExpectedParams(params, grads))
				assertFloat64Values(stateDict.M, scaled(grads, 0.1))
				assertFloat64Values(stateDict.V, absValues(grads))
			})

			Convey("It should preserve optimizer state across repeated steps", func() {
				stateDict := adaMaxState([]float64{3.0}, []float64{6.0})
				opt := NewAdaMax()

				for range 2000 {
					updated, err := opt.Step(stateDict)
					So(err, ShouldBeNil)

					params := cloneFloat64(updated.Out)
					stateDict.
						WithParams(updated.Out).
						WithGrads([]float64{2 * params[0]})
				}

				params := cloneFloat64(stateDict.Out)

				So(stateDict.Step, ShouldEqual, 2000)
				So(stdmath.Abs(params[0]), ShouldBeLessThan, 0.5)
			})

			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := adamParityVectors(parameterCount)
					stateDict := adaMaxState(params, grads)
					opt := NewAdaMax()

					updated, err := opt.Step(stateDict)

					So(err, ShouldBeNil)
					assertFloat64Values(updated.Out, adaMaxExpectedParams(params, grads))
					assertFloat64Values(updated.M, scaled(grads, 0.1))
					assertFloat64Values(updated.V, absValues(grads))
				}
			})

			Convey("It should reject mismatched params and gradients", func() {
				stateDict := adaMaxState([]float64{1.0, 2.0}, []float64{1.0})
				opt := NewAdaMax()

				updated, err := opt.Step(stateDict)

				So(err, ShouldNotBeNil)
				So(updated, ShouldBeNil)
				So(err.Error(), ShouldContainSubstring, "length mismatch")
			})
		})
	})
}

func BenchmarkAdaMax_Step(b *testing.B) {
	for _, size := range []int{1 << 10, 1 << 20} {
		size := size

		b.Run(benchmarkName(size), func(b *testing.B) {
			params, grads := benchmarkVectors(size)
			stateDict := adaMaxState(params, grads)
			opt := NewAdaMax()

			for b.Loop() {
				updated, err := opt.Step(stateDict)

				if err != nil {
					b.Fatalf("Step failed: %v", err)
				}

				stateDict.WithParams(updated.Out)
			}
		})
	}
}

func adaMaxState(params, grads []float64) *state.Dict {
	return state.NewDict().
		WithLR(0.01).
		WithBeta1(0.9).
		WithBeta2(0.999).
		WithEps(1e-8).
		WithParams(params).
		WithGrads(grads)
}

func adaMaxExpectedParams(params, grads []float64) []float64 {
	expected := make([]float64, len(params))
	lrT := 0.01 / (1 - 0.9)

	for index := range params {
		m := 0.1 * grads[index]
		u := stdmath.Abs(grads[index])
		expected[index] = params[index] - lrT*m/(u+1e-8)
	}

	return expected
}

func absValues(values []float64) []float64 {
	out := make([]float64, len(values))

	for index, value := range values {
		out[index] = stdmath.Abs(value)
	}

	return out
}
