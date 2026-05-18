//go:build darwin && cgo

package metal

import (
	"context"
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestActiveInferenceOps_FreeEnergyTensor(test *testing.T) {
	Convey("Given resident Metal active-inference free-energy inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		activeOps := activeInferenceOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar reference at contract sizes", func() {
			for _, dimension := range metalContractSizes() {
				mean, logSigma := activeInferenceGaussianInputs(dimension)
				output, err := activeOps.FreeEnergyTensor(
					uploadMetalTensorForTest(test, tensorBackend, activeShape(test, dimension), mean),
					uploadMetalTensorForTest(test, tensorBackend, activeShape(test, dimension), logSigma),
					activeShape(test, 1),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(
					values,
					[]float64{referenceActiveFreeEnergy(mean, logSigma)},
					math.Max(1e-4, 1e-7*float64(dimension)),
				)
			}
		})
	})
}

func TestActiveInferenceOps_BeliefUpdateTensor(test *testing.T) {
	Convey("Given resident Metal active-inference belief-update inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		activeOps := activeInferenceOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar update reference", func() {
			for _, dimension := range metalContractSizes() {
				mean, logSigma := activeInferenceGaussianInputs(dimension)
				predictionError := activeInferenceSequence(dimension, 0.019, -0.14)
				learningRate := float32(0.0125)
				output, err := activeOps.BeliefUpdateTensor(
					uploadMetalTensorForTest(test, tensorBackend, activeShape(test, dimension), mean),
					uploadMetalTensorForTest(test, tensorBackend, activeShape(test, dimension), logSigma),
					uploadMetalTensorForTest(test, tensorBackend, activeShape(test, dimension), predictionError),
					activeShape(test, 2*dimension),
					learningRate,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(
					values,
					referenceActiveBeliefUpdate(mean, logSigma, predictionError, learningRate),
					1e-5,
				)
			}
		})
	})
}

func TestActiveInferenceOps_PrecisionWeightTensor(test *testing.T) {
	Convey("Given resident Metal active-inference precision-weight inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		activeOps := activeInferenceOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar precision reference", func() {
			for _, dimension := range metalContractSizes() {
				predictionError := activeInferenceSequence(dimension, 0.015, -0.2)
				logPrecision := activeInferenceSequence(dimension, 0.004, -0.03)
				output, err := activeOps.PrecisionWeightTensor(
					uploadMetalTensorForTest(test, tensorBackend, activeShape(test, dimension), predictionError),
					uploadMetalTensorForTest(test, tensorBackend, activeShape(test, dimension), logPrecision),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceActivePrecisionWeight(predictionError, logPrecision), 1e-5)
			}
		})
	})
}

func TestActiveInferenceOps_ExpectedFreeEnergyTensor(test *testing.T) {
	Convey("Given resident Metal active-inference expected-free-energy inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		activeOps := activeInferenceOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar expected free energy reference", func() {
			for _, dimension := range metalContractSizes() {
				policyCount := 3
				probabilities := activeInferenceOutcomeInputs(dimension, policyCount)
				output, err := activeOps.ExpectedFreeEnergyTensor(
					uploadMetalTensorForTest(
						test,
						tensorBackend,
						activeShape(test, dimension*policyCount),
						probabilities,
					),
					activeShape(test, policyCount),
					dimension,
					policyCount,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(
					values,
					referenceActiveExpectedFreeEnergy(probabilities, dimension, policyCount),
					1e-4,
				)
			}
		})
	})
}

func TestTensorBackend_applyActiveInferenceGraph(test *testing.T) {
	Convey("Given Metal active-inference graph execution", test, func() {
		Convey("It should keep active-inference graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, targets, expectedBytes, expected := activeInferenceGraph(test)

			before := tensorBackend.runtime.Metrics()
			results, err := NewRunnerWithBackend(tensorBackend).Execute(
				context.Background(),
				graph,
				targets,
			)
			after := tensorBackend.runtime.Metrics()
			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, len(targets))
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, expectedBytes)
			assertActiveInferenceGraphOutputs(results, expected)
		})
	})
}

func BenchmarkActiveInferenceOps_FreeEnergyTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, activeOps := activeBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	mean, logSigma := activeInferenceGaussianInputs(8192)
	meanTensor := uploadMetalTensor(tensorBackend, activeBenchmarkShape(benchmark, 8192), mean)
	logSigmaTensor := uploadMetalTensor(tensorBackend, activeBenchmarkShape(benchmark, 8192), logSigma)
	outputShape := activeBenchmarkShape(benchmark, 1)
	defer closeBenchmarkTensors(meanTensor, logSigmaTensor)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := activeOps.FreeEnergyTensor(meanTensor, logSigmaTensor, outputShape)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkActiveInferenceOps_BeliefUpdateTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, activeOps := activeBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	mean, logSigma := activeInferenceGaussianInputs(8192)
	predictionError := activeInferenceSequence(8192, 0.019, -0.14)
	meanTensor := uploadMetalTensor(tensorBackend, activeBenchmarkShape(benchmark, 8192), mean)
	logSigmaTensor := uploadMetalTensor(tensorBackend, activeBenchmarkShape(benchmark, 8192), logSigma)
	errorTensor := uploadMetalTensor(tensorBackend, activeBenchmarkShape(benchmark, 8192), predictionError)
	outputShape := activeBenchmarkShape(benchmark, 16384)
	defer closeBenchmarkTensors(meanTensor, logSigmaTensor, errorTensor)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := activeOps.BeliefUpdateTensor(
			meanTensor,
			logSigmaTensor,
			errorTensor,
			outputShape,
			0.0125,
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkActiveInferenceOps_PrecisionWeightTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, activeOps := activeBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	predictionError := activeInferenceSequence(8192, 0.015, -0.2)
	logPrecision := activeInferenceSequence(8192, 0.004, -0.03)
	errorTensor := uploadMetalTensor(tensorBackend, activeBenchmarkShape(benchmark, 8192), predictionError)
	precisionTensor := uploadMetalTensor(tensorBackend, activeBenchmarkShape(benchmark, 8192), logPrecision)
	defer closeBenchmarkTensors(errorTensor, precisionTensor)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := activeOps.PrecisionWeightTensor(errorTensor, precisionTensor)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkActiveInferenceOps_ExpectedFreeEnergyTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, activeOps := activeBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	policyCount := 3
	probabilities := activeInferenceOutcomeInputs(8192, policyCount)
	qTensor := uploadMetalTensor(tensorBackend, activeBenchmarkShape(benchmark, len(probabilities)), probabilities)
	outputShape := activeBenchmarkShape(benchmark, policyCount)
	defer closeBenchmarkTensors(qTensor)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := activeOps.ExpectedFreeEnergyTensor(qTensor, outputShape, 8192, policyCount)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func activeInferenceOpsForTest(test testing.TB, tensorBackend *TensorBackend) *ActiveInferenceOps {
	test.Helper()

	activeOps, err := tensorBackend.activeInference()
	So(err, ShouldBeNil)

	return activeOps
}
