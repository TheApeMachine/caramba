//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalPredictiveCodingOps_PredictionTensor(test *testing.T) {
	Convey("Given resident Metal predictive-coding prediction inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		predictiveOps := predictiveOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar prediction reference", func() {
			for _, inFeatures := range metalContractSizes() {
				outFeatures := 3
				weights := predictiveWeights(outFeatures, inFeatures)
				representation := predictiveVector(inFeatures, 0.013, -0.2)
				outputShape := predictiveShape(test, outFeatures)
				output, err := predictiveOps.PredictionTensor(
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, len(weights)), weights),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, inFeatures), representation),
					outputShape,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referencePredictivePrediction(weights, representation, outFeatures, inFeatures), 1e-4)
			}
		})
	})
}

func TestMetalPredictiveCodingOps_PredictionErrorTensor(test *testing.T) {
	Convey("Given resident Metal predictive-coding error inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		predictiveOps := predictiveOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar error reference", func() {
			for _, dimension := range metalContractSizes() {
				observation := predictiveVector(dimension, 0.011, 0.3)
				prediction := predictiveVector(dimension, 0.009, -0.1)
				precision := predictiveVector(dimension, 0.003, 0.8)
				output, err := predictiveOps.PredictionErrorTensor(
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, dimension), observation),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, dimension), prediction),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, dimension), precision),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referencePredictiveError(observation, prediction, precision), 1e-6)
			}
		})
	})
}

func TestMetalPredictiveCodingOps_UpdateRepresentationTensor(test *testing.T) {
	Convey("Given resident Metal predictive-coding representation-update inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		predictiveOps := predictiveOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar representation reference", func() {
			for _, inFeatures := range metalContractSizes() {
				outFeatures := 3
				representation := predictiveVector(inFeatures, 0.013, -0.2)
				weights := predictiveWeights(outFeatures, inFeatures)
				lowerError := predictiveVector(outFeatures, 0.017, -0.05)
				selfError := predictiveVector(inFeatures, 0.007, 0.04)
				learningRate := []float64{0.025}
				output, err := predictiveOps.UpdateRepresentationTensor(
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, inFeatures), representation),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, len(weights)), weights),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, outFeatures), lowerError),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, inFeatures), selfError),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, 1), learningRate),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(
					values,
					referencePredictiveUpdateRepresentation(
						representation, weights, lowerError, selfError, learningRate[0], outFeatures, inFeatures,
					),
					1e-5,
				)
			}
		})
	})
}

func TestMetalPredictiveCodingOps_UpdateWeightsTensor(test *testing.T) {
	Convey("Given resident Metal predictive-coding weight-update inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		predictiveOps := predictiveOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar weight reference", func() {
			for _, inFeatures := range metalContractSizes() {
				outFeatures := 3
				weights := predictiveWeights(outFeatures, inFeatures)
				errorValues := predictiveVector(outFeatures, 0.017, -0.05)
				representation := predictiveVector(inFeatures, 0.013, -0.2)
				learningRate := []float64{0.025}
				output, err := predictiveOps.UpdateWeightsTensor(
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, len(weights)), weights),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, outFeatures), errorValues),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, inFeatures), representation),
					predictiveShape(test, len(weights)),
					uploadMetalTensorForTest(test, tensorBackend, predictiveShape(test, 1), learningRate),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(
					values,
					referencePredictiveUpdateWeights(weights, errorValues, representation, learningRate[0], outFeatures, inFeatures),
					1e-6,
				)
			}
		})
	})
}

func TestTensorBackend_applyPredictiveCodingGraph(test *testing.T) {
	Convey("Given Metal predictive-coding graph execution", test, func() {
		Convey("It should keep predictive-coding graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, targets, expectedBytes, expected := predictiveGraph(test)

			before := tensorBackend.runtime.Metrics()
			results, err := NewRunnerWithBackend(tensorBackend).Execute(context.Background(), graph, targets)
			after := tensorBackend.runtime.Metrics()
			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, len(targets))
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, expectedBytes)
			assertPredictiveGraphOutputs(results, expected)
		})
	})
}

func BenchmarkMetalPredictiveCodingOps_PredictionTensor(benchmark *testing.B) {
	benchmarkPredictive(benchmark, "prediction")
}

func BenchmarkMetalPredictiveCodingOps_PredictionErrorTensor(benchmark *testing.B) {
	benchmarkPredictive(benchmark, "prediction_error")
}

func BenchmarkMetalPredictiveCodingOps_UpdateRepresentationTensor(benchmark *testing.B) {
	benchmarkPredictive(benchmark, "update_representation")
}

func BenchmarkMetalPredictiveCodingOps_UpdateWeightsTensor(benchmark *testing.B) {
	benchmarkPredictive(benchmark, "update_weights")
}

func predictiveOpsForTest(test testing.TB, tensorBackend *TensorBackend) *MetalPredictiveCodingOps {
	test.Helper()

	predictiveOps, err := tensorBackend.predictiveCoding()
	So(err, ShouldBeNil)

	return predictiveOps
}
