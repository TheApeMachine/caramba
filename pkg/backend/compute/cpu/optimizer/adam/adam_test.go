package adam

import (
	stdmath "math"
	"os"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestAdam_Step(t *testing.T) {
	Convey("Given an Adam optimizer", t, func() {
		Convey("Step", func() {
			cases := []struct {
				name   string
				params []float64
				grads  []float64
			}{
				{
					name:   "It should update params and moments on the first Adam step",
					params: []float64{1.0, -2.0, 0.5},
					grads:  []float64{0.1, -0.2, 0.05},
				},
				{
					name:   "It should ignore decoupled weight decay reserved for AdamW",
					params: []float64{1.0, -2.0, 0.5},
					grads:  []float64{0.0, 0.0, 0.0},
				},
			}

			for _, testCase := range cases {
				testCase := testCase

				Convey(testCase.name, func() {
					stateDict := adamState(testCase.params, testCase.grads, 0.1)
					opt := NewAdam()

					updated, err := opt.Step(stateDict)

					So(err, ShouldBeNil)
					So(stateDict.Step, ShouldEqual, 1)
					assertFloat64Values(updated.Out, adamExpectedParams(testCase.params, testCase.grads, 0))
					assertFloat64Values(stateDict.Out, adamExpectedParams(testCase.params, testCase.grads, 0))
					assertFloat64Values(stateDict.M, scaled(testCase.grads, 0.1))
					assertFloat64Values(stateDict.V, squaredScaled(testCase.grads, 0.001))
				})
			}

			Convey("It should preserve optimizer state across repeated steps", func() {
				stateDict := adamState([]float64{5.0}, []float64{10.0}, 0)
				opt := NewAdam()

				for range 2000 {
					updated, err := opt.Step(stateDict)
					So(err, ShouldBeNil)

					params := cloneFloat64(updated.Out)
					stateDict.
						WithParams(updated.Out).
						WithGrads(tensor.MustFloat64From([]float64{2 * params[0]}))
				}

				params := cloneFloat64(stateDict.Out)

				So(stateDict.Step, ShouldEqual, 2000)
				So(stdmath.Abs(params[0]), ShouldBeLessThan, 0.1)
			})

			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := adamParityVectors(parameterCount)
					stateDict := adamState(params, grads, 0)
					opt := NewAdam()

					updated, err := opt.Step(stateDict)

					So(err, ShouldBeNil)
					assertFloat64Values(updated.Out, adamExpectedParams(params, grads, 0))
					assertFloat64Values(updated.M, scaled(grads, 0.1))
					assertFloat64Values(updated.V, squaredScaled(grads, 0.001))
				}
			})

			Convey("It should reject mismatched params and gradients", func() {
				stateDict := adamState([]float64{1.0, 2.0}, []float64{1.0}, 0)
				opt := NewAdam()

				updated, err := opt.Step(stateDict)

				So(err, ShouldNotBeNil)
				So(updated, ShouldBeNil)
				So(err.Error(), ShouldContainSubstring, "length mismatch")
			})
		})
	})
}

func TestAdamW_Step(t *testing.T) {
	Convey("Given an AdamW optimizer", t, func() {
		Convey("Step", func() {
			cases := []struct {
				name   string
				params []float64
				grads  []float64
				wd     float64
			}{
				{
					name:   "It should apply decoupled weight decay with zero gradients",
					params: []float64{1.0, -2.0, 0.5},
					grads:  []float64{0.0, 0.0, 0.0},
					wd:     0.1,
				},
				{
					name:   "It should combine Adam moments with decoupled weight decay",
					params: []float64{1.0, -2.0, 0.5},
					grads:  []float64{0.1, -0.2, 0.05},
					wd:     0.1,
				},
			}

			for _, testCase := range cases {
				testCase := testCase

				Convey(testCase.name, func() {
					stateDict := adamState(testCase.params, testCase.grads, testCase.wd)
					opt := NewAdamW()

					updated, err := opt.Step(stateDict)

					So(err, ShouldBeNil)
					So(stateDict.Step, ShouldEqual, 1)
					assertFloat64Values(
						updated.Out,
						adamExpectedParams(testCase.params, testCase.grads, testCase.wd),
					)
					assertFloat64Values(
						stateDict.Out,
						adamExpectedParams(testCase.params, testCase.grads, testCase.wd),
					)
					assertFloat64Values(stateDict.M, scaled(testCase.grads, 0.1))
					assertFloat64Values(stateDict.V, squaredScaled(testCase.grads, 0.001))
				})
			}

			Convey("It should preserve optimizer state across repeated steps", func() {
				stateDict := adamState([]float64{5.0}, []float64{10.0}, 0.01)
				opt := NewAdamW()

				for range 2000 {
					updated, err := opt.Step(stateDict)
					So(err, ShouldBeNil)

					params := cloneFloat64(updated.Out)
					stateDict.
						WithParams(updated.Out).
						WithGrads(tensor.MustFloat64From([]float64{2 * params[0]}))
				}

				params := cloneFloat64(stateDict.Out)

				So(stateDict.Step, ShouldEqual, 2000)
				So(stdmath.Abs(params[0]), ShouldBeLessThan, 0.1)
			})

			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, parameterCount := range []int{1, 7, 64, 1024, 8192} {
					params, grads := adamParityVectors(parameterCount)
					stateDict := adamState(params, grads, 0.01)
					opt := NewAdamW()

					updated, err := opt.Step(stateDict)

					So(err, ShouldBeNil)
					assertFloat64Values(updated.Out, adamExpectedParams(params, grads, 0.01))
					assertFloat64Values(updated.M, scaled(grads, 0.1))
					assertFloat64Values(updated.V, squaredScaled(grads, 0.001))
				}
			})

			Convey("It should reject mismatched params and gradients", func() {
				stateDict := adamState([]float64{1.0, 2.0}, []float64{1.0}, 0.1)
				opt := NewAdamW()

				updated, err := opt.Step(stateDict)

				So(err, ShouldNotBeNil)
				So(updated, ShouldBeNil)
				So(err.Error(), ShouldContainSubstring, "length mismatch")
			})
		})
	})
}

func TestAdam_SIMDSourceSeparation(t *testing.T) {
	Convey("Given the Adam optimizer SIMD sources", t, func() {
		Convey("It should keep SSE, AdamW, and AdaMax implementations explicit", func() {
			cases := []struct {
				name      string
				file      string
				symbol    string
				forbidden []string
			}{
				{
					name:   "It should implement Adam SSE2 in its own file",
					file:   "adam_sse2_amd64.s",
					symbol: "TEXT ·adamStepSSE2(SB)",
				},
				{
					name:      "It should implement AdamW AVX2 without jumping into Adam",
					file:      "adamw_avx2_amd64.s",
					symbol:    "TEXT ·adamwStepAVX2(SB)",
					forbidden: []string{"JMP ·adamStep", "CALL ·adamStep"},
				},
				{
					name:      "It should implement AdamW SSE2 without jumping into Adam",
					file:      "adamw_sse2_amd64.s",
					symbol:    "TEXT ·adamwStepSSE2(SB)",
					forbidden: []string{"JMP ·adamStep", "CALL ·adamStep"},
				},
				{
					name:      "It should implement AdamW NEON without jumping into Adam",
					file:      "adamw_neon_arm64.s",
					symbol:    "TEXT ·adamwStepNEON(SB)",
					forbidden: []string{"JMP ·adamStep", "CALL ·adamStep"},
				},
				{
					name:      "It should implement AdaMax SSE2 in its own file",
					file:      "adamax_sse2_amd64.s",
					symbol:    "TEXT ·adamaxStepSSE2(SB)",
					forbidden: []string{"JMP ·adamaxStepAVX2", "CALL ·adamaxStepAVX2"},
				},
			}

			for _, testCase := range cases {
				testCase := testCase

				Convey(testCase.name, func() {
					source, err := os.ReadFile(testCase.file)
					So(err, ShouldBeNil)

					content := string(source)
					So(content, ShouldContainSubstring, testCase.symbol)

					for _, forbidden := range testCase.forbidden {
						So(content, ShouldNotContainSubstring, forbidden)
					}
				})
			}
		})
	})
}

func BenchmarkAdam_Step(b *testing.B) {
	for _, size := range []int{1 << 10, 1 << 20} {
		size := size

		b.Run(benchmarkName(size), func(b *testing.B) {
			params, grads := benchmarkVectors(size)
			stateDict := adamState(params, grads, 0)
			opt := NewAdam()

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

func BenchmarkAdamW_Step(b *testing.B) {
	for _, size := range []int{1 << 10, 1 << 20} {
		size := size

		b.Run(benchmarkName(size), func(b *testing.B) {
			params, grads := benchmarkVectors(size)
			stateDict := adamState(params, grads, 0.01)
			opt := NewAdamW()

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

func adamState(params, grads []float64, wd float64) *state.Dict {
	return state.NewDict().
		WithLR(0.01).
		WithBeta1(0.9).
		WithBeta2(0.999).
		WithEps(1e-8).
		WithWD(wd).
		WithParams(tensor.MustFloat64From(params)).
		WithGrads(tensor.MustFloat64From(grads))
}

func adamExpectedParams(params, grads []float64, wd float64) []float64 {
	expected := make([]float64, len(params))
	lrT := 0.01 * stdmath.Sqrt(1-0.999) / (1 - 0.9)

	for index := range params {
		m := 0.1 * grads[index]
		v := 0.001 * grads[index] * grads[index]
		expected[index] = params[index] - lrT*m/(stdmath.Sqrt(v)+1e-8) - 0.01*wd*params[index]
	}

	return expected
}

func benchmarkName(size int) string {
	switch size {
	case 1 << 10:
		return "1K"
	case 1 << 20:
		return "1M"
	default:
		return "custom"
	}
}

func benchmarkVectors(size int) ([]float64, []float64) {
	params := make([]float64, size)
	grads := make([]float64, size)

	for index := range params {
		params[index] = 1e-3
		grads[index] = 1e-4
	}

	return params, grads
}

func adamParityVectors(size int) ([]float64, []float64) {
	params := make([]float64, size)
	grads := make([]float64, size)

	for index := range params {
		params[index] = float64(index%17-8) * 0.125
		grads[index] = float64(index%11-5) * 0.0625
	}

	return params, grads
}

func scaled(values []float64, scale float64) []float64 {
	out := make([]float64, len(values))

	for index, value := range values {
		out[index] = scale * value
	}

	return out
}

func squaredScaled(values []float64, scale float64) []float64 {
	out := make([]float64, len(values))

	for index, value := range values {
		out[index] = scale * value * value
	}

	return out
}

func assertFloat64Values(actual []float64, expected []float64) {
	So(actual, ShouldHaveLength, len(expected))

	for index, value := range actual {
		So(value, ShouldAlmostEqual, expected[index], 1e-12)
	}
}

func cloneFloat64(values []float64) []float64 {
	return append([]float64(nil), values...)
}
