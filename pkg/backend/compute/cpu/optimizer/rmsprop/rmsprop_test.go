package rmsprop

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestRMSProp_Step(test *testing.T) {
	Convey("Given an RMSProp optimizer", test, func() {
		Convey("Step", func() {
			Convey("It should match scalar plain parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := rmspropParityState(parameterCount)
					stateDict := rmspropState(params, grads, false, 0)
					initialV := append([]float64(nil), stateDict.V...)

					updated, err := NewRMSProp().Step(stateDict)

					So(err, ShouldBeNil)

					for parameterIndex := range parameterCount {
						expected := rmspropReferenceStep(rmspropReferenceInput{
							param:    params[parameterIndex],
							grad:     grads[parameterIndex],
							v:        initialV[parameterIndex],
							centered: false,
						})

						So(updated.Out[parameterIndex], ShouldAlmostEqual, expected.out, 1e-12)
						So(updated.V[parameterIndex], ShouldAlmostEqual, expected.v, 1e-12)
					}
				}
			})

			Convey("It should match scalar centered parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := rmspropParityState(parameterCount)
					stateDict := rmspropState(params, grads, true, 0)
					initialV := append([]float64(nil), stateDict.V...)
					initialGradAvg := append([]float64(nil), stateDict.GradAvg...)

					updated, err := NewRMSProp().Step(stateDict)

					So(err, ShouldBeNil)

					for parameterIndex := range parameterCount {
						expected := rmspropReferenceStep(rmspropReferenceInput{
							param:    params[parameterIndex],
							grad:     grads[parameterIndex],
							v:        initialV[parameterIndex],
							gradAvg:  initialGradAvg[parameterIndex],
							centered: true,
						})

						So(updated.Out[parameterIndex], ShouldAlmostEqual, expected.out, 1e-12)
						So(updated.V[parameterIndex], ShouldAlmostEqual, expected.v, 1e-12)
						So(updated.GradAvg[parameterIndex], ShouldAlmostEqual, expected.gradAvg, 1e-12)
					}
				}
			})

			Convey("It should match scalar momentum parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := rmspropParityState(parameterCount)
					stateDict := rmspropState(params, grads, false, 0.75)
					initialV := append([]float64(nil), stateDict.V...)
					initialBuf := append([]float64(nil), stateDict.Buf...)

					updated, err := NewRMSProp().Step(stateDict)

					So(err, ShouldBeNil)

					for parameterIndex := range parameterCount {
						expected := rmspropReferenceStep(rmspropReferenceInput{
							param:    params[parameterIndex],
							grad:     grads[parameterIndex],
							v:        initialV[parameterIndex],
							buf:      initialBuf[parameterIndex],
							momentum: 0.75,
						})

						So(updated.Out[parameterIndex], ShouldAlmostEqual, expected.out, 1e-12)
						So(updated.V[parameterIndex], ShouldAlmostEqual, expected.v, 1e-12)
						So(updated.Buf[parameterIndex], ShouldAlmostEqual, expected.buf, 1e-12)
					}
				}
			})

			Convey("It should match scalar centered momentum parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := rmspropParityState(parameterCount)
					stateDict := rmspropState(params, grads, true, 0.75)
					initialV := append([]float64(nil), stateDict.V...)
					initialGradAvg := append([]float64(nil), stateDict.GradAvg...)
					initialBuf := append([]float64(nil), stateDict.Buf...)

					updated, err := NewRMSProp().Step(stateDict)

					So(err, ShouldBeNil)

					for parameterIndex := range parameterCount {
						expected := rmspropReferenceStep(rmspropReferenceInput{
							param:    params[parameterIndex],
							grad:     grads[parameterIndex],
							v:        initialV[parameterIndex],
							gradAvg:  initialGradAvg[parameterIndex],
							buf:      initialBuf[parameterIndex],
							centered: true,
							momentum: 0.75,
						})

						So(updated.Out[parameterIndex], ShouldAlmostEqual, expected.out, 1e-12)
						So(updated.V[parameterIndex], ShouldAlmostEqual, expected.v, 1e-12)
						So(updated.GradAvg[parameterIndex], ShouldAlmostEqual, expected.gradAvg, 1e-12)
						So(updated.Buf[parameterIndex], ShouldAlmostEqual, expected.buf, 1e-12)
					}
				}
			})
		})
	})
}

func BenchmarkRMSProp_Step(benchmark *testing.B) {
	params, grads := rmspropParityState(1 << 20)
	stateDict := rmspropState(params, grads, true, 0.75)
	optimizer := NewRMSProp()

	for benchmark.Loop() {
		updated, err := optimizer.Step(stateDict)

		if err != nil {
			benchmark.Fatalf("Step failed: %v", err)
		}

		stateDict.WithParams(updated.Out)
	}
}

func rmspropState(
	params []float64,
	grads []float64,
	centered bool,
	momentum float64,
) *state.Dict {
	stateDict := state.NewDict().
		WithLR(0.025).
		WithAlpha(0.9).
		WithEps(1e-6).
		WithWD(0.01).
		WithCentered(centered).
		WithMomentum(momentum).
		WithParams(params).
		WithGrads(grads)

	stateDict.V = make([]float64, len(params))
	stateDict.Buf = make([]float64, len(params))
	stateDict.GradAvg = make([]float64, len(params))

	return stateDict
}

func rmspropParityState(parameterCount int) ([]float64, []float64) {
	params := make([]float64, parameterCount)
	grads := make([]float64, parameterCount)

	for parameterIndex := range parameterCount {
		params[parameterIndex] = float64(parameterIndex%17-8) * 0.125
		grads[parameterIndex] = float64(parameterIndex%11-5) * 0.0625
	}

	return params, grads
}

type rmspropReferenceInput struct {
	param    float64
	grad     float64
	v        float64
	gradAvg  float64
	buf      float64
	centered bool
	momentum float64
}

type rmspropReferenceOutput struct {
	out     float64
	v       float64
	gradAvg float64
	buf     float64
}

func rmspropReferenceStep(input rmspropReferenceInput) rmspropReferenceOutput {
	learningRate := 0.025
	alpha := 0.9
	oneMinusAlpha := 1 - alpha
	epsilon := 1e-6
	weightDecay := 0.01

	grad := input.grad + weightDecay*input.param
	updatedV := alpha*input.v + oneMinusAlpha*grad*grad
	updatedGradAvg := input.gradAvg

	denominatorBase := updatedV

	if input.centered {
		updatedGradAvg = alpha*input.gradAvg + oneMinusAlpha*grad
		denominatorBase -= updatedGradAvg * updatedGradAvg
	}

	denominator := stdmath.Sqrt(denominatorBase) + epsilon
	updatedBuf := input.buf
	updatedOut := input.param

	if input.momentum != 0 {
		updatedBuf = input.momentum*input.buf + grad/denominator
		updatedOut -= learningRate * updatedBuf
		return rmspropReferenceOutput{
			out:     updatedOut,
			v:       updatedV,
			gradAvg: updatedGradAvg,
			buf:     updatedBuf,
		}
	}

	updatedOut -= learningRate * grad / denominator

	return rmspropReferenceOutput{
		out:     updatedOut,
		v:       updatedV,
		gradAvg: updatedGradAvg,
		buf:     updatedBuf,
	}
}
