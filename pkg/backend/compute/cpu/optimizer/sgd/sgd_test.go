package sgd

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestSGD_Step(t *testing.T) {
	Convey("Given an SGD optimizer", t, func() {
		Convey("Step", func() {
			Convey("It should move parameters in the negative gradient direction", func() {
				opt := NewSGD()
				stateDict := sgdState([]float64{1.0, 2.0, 3.0}, []float64{1.0, 1.0, 1.0}, 0.1, 0, 0, false)

				updated, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				So(updated.Out[0], ShouldAlmostEqual, 0.9)
				So(updated.Out[1], ShouldAlmostEqual, 1.9)
				So(updated.Out[2], ShouldAlmostEqual, 2.9)
			})

			Convey("It should accumulate momentum across steps", func() {
				opt := NewSGD()
				params := []float64{1.0, 1.0}
				grads := []float64{1.0, 1.0}
				stateDict := sgdState(params, grads, 0.1, 0.9, 0, false)
				p1, err := opt.Step(stateDict)
				So(err, ShouldBeNil)
				firstOut := append([]float64(nil), p1.Out...)
				stateDict.WithParams(firstOut).WithGrads(grads)

				p2, err := opt.Step(stateDict)
				So(err, ShouldBeNil)

				delta1 := params[0] - firstOut[0]
				delta2 := firstOut[0] - p2.Out[0]
				So(delta2, ShouldBeGreaterThan, delta1)
			})

			Convey("It should apply weight decay", func() {
				opt := NewSGD()
				stateDict := sgdState([]float64{2.0}, []float64{0.0}, 0.1, 0, 0.1, false)

				updated, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				So(updated.Out[0], ShouldAlmostEqual, 1.98)
			})

			Convey("It should not mutate the input params slice", func() {
				opt := NewSGD()
				params := []float64{5.0, 5.0}
				stateDict := sgdState(params, []float64{1.0, 1.0}, 0.1, 0, 0, false)
				_, err := opt.Step(stateDict)

				So(err, ShouldBeNil)
				So(params[0], ShouldEqual, 5.0)
			})

			Convey("It should match scalar parity across SIMD tails and momentum modes", func() {
				opt := NewSGD()

				for _, length := range []int{1, 2, 3, 4, 7, 64, 1024} {
					for _, momentum := range []float64{0, 0.72} {
						for _, nesterov := range []bool{false, true} {
							params := sgdPattern(length, 0.17, 0.031)
							grads := sgdPattern(length, -0.11, 0.023)
							velocity := sgdPattern(length, 0.013, 0.007)
							stateDict := sgdState(params, grads, 0.0125, momentum, 0.015, nesterov).
								WithM(append([]float64(nil), velocity...))
							expectedOut, expectedVelocity := sgdReferenceStep(
								params, grads, velocity, 0.0125, 0.015, momentum, nesterov,
							)

							updated, err := opt.Step(stateDict)

							So(err, ShouldBeNil)
							So(updated.Out, ShouldHaveLength, len(expectedOut))

							for index := range expectedOut {
								So(updated.Out[index], ShouldAlmostEqual, expectedOut[index], 1e-10)
							}

							if momentum == 0 {
								continue
							}

							for index := range expectedVelocity {
								So(updated.M[index], ShouldAlmostEqual, expectedVelocity[index], 1e-10)
							}
						}
					}
				}
			})
		})
	})
}

func BenchmarkSGD_Step(b *testing.B) {
	opt := NewSGD()
	n := 1 << 20
	params := make([]float64, n)
	grads := make([]float64, n)
	for i := range params {
		params[i] = float64(i) * 1e-4
		grads[i] = float64(i) * 1e-5
	}
	stateDict := sgdState(params, grads, 0.01, 0.9, 1e-4, false)
	b.ResetTimer()
	for b.Loop() {
		updated, err := opt.Step(stateDict)
		if err != nil {
			b.Fatalf("Step failed: %v", err)
		}
		stateDict.WithParams(updated.Out)
	}
}

func sgdState(params, grads []float64, lr, momentum, wd float64, nesterov bool) *state.Dict {
	return state.NewDict().
		WithLR(lr).
		WithMomentum(momentum).
		WithWD(wd).
		WithNesterov(nesterov).
		WithParams(params).
		WithGrads(grads)
}

func sgdPattern(length int, offset float64, step float64) []float64 {
	values := make([]float64, length)

	for index := range values {
		sign := 1.0

		if index%2 != 0 {
			sign = -1.0
		}

		values[index] = sign*(offset+step*float64(index%11)) + 0.0007*float64(index/11)
	}

	return values
}

func sgdReferenceStep(
	params, grads, velocity []float64,
	lr, wd, momentum float64,
	nesterov bool,
) ([]float64, []float64) {
	out := make([]float64, len(params))
	nextVelocity := append([]float64(nil), velocity...)

	for index, param := range params {
		grad := grads[index] + wd*param

		if momentum == 0 {
			out[index] = param - lr*grad
			continue
		}

		nextVelocity[index] = momentum*nextVelocity[index] + grad
		update := nextVelocity[index]

		if nesterov {
			update = grad + momentum*nextVelocity[index]
		}

		out[index] = param - lr*update
	}

	return out, nextVelocity
}
