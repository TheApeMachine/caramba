//go:build darwin && cgo

package metal

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	cpuai "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/active_inference"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type metalTestSkipper interface {
	Helper()
	Skipf(string, ...any)
}

func metallibPathOrSkip(test metalTestSkipper, name string) string {
	test.Helper()

	_, file, _, ok := runtime.Caller(0)

	if !ok {
		test.Skipf("runtime.Caller failed")
	}

	path := filepath.Join(filepath.Dir(file), name)

	if _, err := os.Stat(path); err != nil {
		test.Skipf("missing %s; run `make build` in repo root", path)
	}

	return path
}

func TestActiveInferenceOps_FreeEnergy(test *testing.T) {
	lib := metallibPathOrSkip(test, "active_inference.metallib")

	Convey("Given Metal free energy", test, func() {
		metalOps := newActiveInferenceOpsForTest(test, lib)

		Convey("It should match CPU state-dict execution at contract sizes", func() {
			for _, dimension := range metalContractSizes() {
				mean, logSigma := activeInferenceGaussianInputs(dimension)
				expectedState, err := cpuai.NewFreeEnergy().Forward(
					state.NewDict().WithShape([]int{dimension}).WithInputs(mean, logSigma),
				)
				actual, metalErr := metalOps.FreeEnergy([]int{dimension}, mean, logSigma)

				So(err, ShouldBeNil)
				So(metalErr, ShouldBeNil)
				So(actual, ShouldHaveLength, 1)
				assertMetalMaxDiff(
					actual,
					expectedState.Out,
					math.Max(5e-3, 5e-5*float64(dimension)),
				)
			}
		})
	})
}

func TestActiveInferenceOps_BeliefUpdate(test *testing.T) {
	lib := metallibPathOrSkip(test, "active_inference.metallib")

	Convey("Given Metal belief update", test, func() {
		metalOps := newActiveInferenceOpsForTest(test, lib)

		Convey("It should match CPU state-dict execution at contract sizes", func() {
			for _, dimension := range metalContractSizes() {
				mean, logSigma := activeInferenceGaussianInputs(dimension)
				predictionError := activeInferenceSequence(dimension, 0.019, -0.14)
				expectedState, err := cpuai.NewBeliefUpdate().Forward(
					state.NewDict().WithShape([]int{dimension, 125}).WithInputs(
						mean,
						logSigma,
						predictionError,
					),
				)
				actual, metalErr := metalOps.BeliefUpdate(
					[]int{dimension, 125},
					mean,
					logSigma,
					predictionError,
				)

				So(err, ShouldBeNil)
				So(metalErr, ShouldBeNil)
				assertMetalMaxDiff(actual, expectedState.Out, 1e-5)
			}
		})
	})
}

func TestActiveInferenceOps_PrecisionWeight(test *testing.T) {
	lib := metallibPathOrSkip(test, "active_inference.metallib")

	Convey("Given Metal precision weighting", test, func() {
		metalOps := newActiveInferenceOpsForTest(test, lib)

		Convey("It should match CPU state-dict execution at contract sizes", func() {
			for _, dimension := range metalContractSizes() {
				predictionError := activeInferenceSequence(dimension, 0.015, -0.2)
				logPrecision := activeInferenceSequence(dimension, 0.004, -0.03)
				expectedState, err := cpuai.NewPrecisionWeight().Forward(
					state.NewDict().WithShape([]int{dimension}).WithInputs(
						predictionError,
						logPrecision,
					),
				)
				actual, metalErr := metalOps.PrecisionWeight(
					[]int{dimension},
					predictionError,
					logPrecision,
				)

				So(err, ShouldBeNil)
				So(metalErr, ShouldBeNil)
				assertMetalMaxDiff(actual, expectedState.Out, 1e-5)
			}
		})
	})
}

func TestActiveInferenceOps_ExpectedFreeEnergy(test *testing.T) {
	lib := metallibPathOrSkip(test, "active_inference.metallib")

	Convey("Given Metal expected free energy", test, func() {
		metalOps := newActiveInferenceOpsForTest(test, lib)

		Convey("It should clamp probabilities and match CPU state-dict execution at contract sizes", func() {
			for _, dimension := range metalContractSizes() {
				outcomeCount := 3
				probabilities := activeInferenceOutcomeInputs(dimension, outcomeCount)
				expectedState, err := cpuai.NewExpectedFreeEnergy().Forward(
					state.NewDict().WithShape([]int{dimension, outcomeCount}).WithInput(probabilities),
				)
				actual, metalErr := metalOps.ExpectedFreeEnergy(
					[]int{dimension, outcomeCount},
					probabilities,
				)

				So(err, ShouldBeNil)
				So(metalErr, ShouldBeNil)
				assertMetalMaxDiff(
					actual,
					expectedState.Out,
					math.Max(1e-4, 1e-5*float64(dimension)),
				)
			}
		})
	})
}

func BenchmarkActiveInferenceOps_FreeEnergy(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "active_inference.metallib")
	metalOps := newActiveInferenceOpsForBenchmark(benchmark, lib)
	mean, logSigma := activeInferenceGaussianInputs(8192)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := metalOps.FreeEnergy([]int{8192}, mean, logSigma); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkActiveInferenceOps_BeliefUpdate(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "active_inference.metallib")
	metalOps := newActiveInferenceOpsForBenchmark(benchmark, lib)
	mean, logSigma := activeInferenceGaussianInputs(8192)
	predictionError := activeInferenceSequence(8192, 0.019, -0.14)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := metalOps.BeliefUpdate(
			[]int{8192, 125},
			mean,
			logSigma,
			predictionError,
		); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkActiveInferenceOps_PrecisionWeight(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "active_inference.metallib")
	metalOps := newActiveInferenceOpsForBenchmark(benchmark, lib)
	predictionError := activeInferenceSequence(8192, 0.015, -0.2)
	logPrecision := activeInferenceSequence(8192, 0.004, -0.03)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := metalOps.PrecisionWeight(
			[]int{8192},
			predictionError,
			logPrecision,
		); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkActiveInferenceOps_ExpectedFreeEnergy(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "active_inference.metallib")
	metalOps := newActiveInferenceOpsForBenchmark(benchmark, lib)
	probabilities := activeInferenceOutcomeInputs(8192, 3)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := metalOps.ExpectedFreeEnergy([]int{8192, 3}, probabilities); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func newActiveInferenceOpsForTest(test *testing.T, lib string) *ActiveInferenceOps {
	test.Helper()

	metalOps, err := NewActiveInferenceOps(lib)
	So(err, ShouldBeNil)

	test.Cleanup(func() {
		if err := metalOps.Close(); err != nil {
			test.Errorf("ActiveInferenceOps.Close: %v", err)
		}
	})

	return metalOps
}

func newActiveInferenceOpsForBenchmark(
	benchmark *testing.B, lib string,
) *ActiveInferenceOps {
	benchmark.Helper()

	metalOps, err := NewActiveInferenceOps(lib)

	if err != nil {
		benchmark.Fatal(err)
	}

	benchmark.Cleanup(func() {
		if err := metalOps.Close(); err != nil {
			benchmark.Fatal(err)
		}
	})

	return metalOps
}

func metalContractSizes() []int {
	return []int{1, 7, 64, 1024, 8192}
}

func activeInferenceGaussianInputs(dimension int) ([]float64, []float64) {
	mean := activeInferenceSequence(dimension, 0.011, -0.07)
	logSigma := activeInferenceSequence(dimension, 0.006, -0.31)

	return mean, logSigma
}

func activeInferenceSequence(dimension int, scale float64, offset float64) []float64 {
	values := make([]float64, dimension)

	for index := range values {
		values[index] = offset + scale*float64(index%23-11)
	}

	return values
}

func activeInferenceOutcomeInputs(dimension int, outcomeCount int) []float64 {
	probabilities := make([]float64, dimension*outcomeCount)

	for index := range probabilities {
		switch index % 11 {
		case 0:
			probabilities[index] = -0.125
		case 1:
			probabilities[index] = 1.125
		default:
			probabilities[index] = 0.05 + 0.035*float64(index%7)
		}
	}

	return probabilities
}
