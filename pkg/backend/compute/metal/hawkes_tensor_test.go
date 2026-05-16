//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalHawkes_IntensityTensor(test *testing.T) {
	Convey("Given resident Metal Hawkes intensity inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		hawkesOps := hawkesOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar reference at event contract sizes", func() {
			for _, eventCount := range metalMathContractSizes {
				processCount := 3
				times := hawkesTimes(eventCount)
				alpha := hawkesAlpha(processCount)
				beta := hawkesBeta(processCount)
				mu := hawkesMu(processCount)
				currentTime := []float64{float64(eventCount)/10 + 1}
				outputShape := hawkesShape(test, processCount)

				output, err := hawkesOps.IntensityTensor(
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, eventCount), times),
					uploadMetalTensorForTest(test, tensorBackend, outputShape, alpha),
					uploadMetalTensorForTest(test, tensorBackend, outputShape, beta),
					uploadMetalTensorForTest(test, tensorBackend, outputShape, mu),
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, 1), currentTime),
					outputShape,
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
					referenceHawkesIntensity(times, alpha, beta, mu, currentTime[0]),
					1e-5,
				)
			}
		})
	})
}

func TestMetalHawkes_KernelMatrixTensor(test *testing.T) {
	Convey("Given resident Metal Hawkes kernel-matrix inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		hawkesOps := hawkesOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar matrix reference", func() {
			for _, eventCount := range []int{1, 7, 64} {
				times := hawkesTimes(eventCount)
				alpha := []float64{0.7}
				beta := []float64{0.4}
				outputShape := hawkesShape(test, eventCount*eventCount)

				output, err := hawkesOps.KernelMatrixTensor(
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, eventCount), times),
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, 1), alpha),
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, 1), beta),
					outputShape,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceHawkesKernelMatrix(times, alpha[0], beta[0]), 1e-5)
			}
		})
	})
}

func TestMetalHawkes_LogLikelihoodTensor(test *testing.T) {
	Convey("Given resident Metal Hawkes log-likelihood inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		hawkesOps := hawkesOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar reduction reference", func() {
			for _, eventCount := range metalMathContractSizes {
				intensities := hawkesIntensities(eventCount)
				integral := []float64{1.25}

				output, err := hawkesOps.LogLikelihoodTensor(
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, eventCount), intensities),
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, 1), integral),
					hawkesShape(test, 1),
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
					[]float64{referenceHawkesLogLikelihood(intensities, integral[0])},
					1e-4,
				)
			}
		})
	})
}

func TestMetalHawkes_SimulateTensor(test *testing.T) {
	Convey("Given resident Metal Hawkes simulation inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		hawkesOps := hawkesOpsForTest(test, tensorBackend)

		Convey("It should match the deterministic scalar simulation reference", func() {
			for _, maxSteps := range []int{1, 7, 64, 1024} {
				processCount := 2
				mu := []float64{1.2, 1.4}
				alpha := []float64{0, 0}
				beta := []float64{0.8, 0.9}
				tMax := []float64{2}
				outputShape := hawkesShape(test, processCount*maxSteps)

				output, err := hawkesOps.SimulateTensor(
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, processCount), mu),
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, processCount), alpha),
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, processCount), beta),
					uploadMetalTensorForTest(test, tensorBackend, hawkesShape(test, 1), tMax),
					outputShape,
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
					referenceHawkesSimulate(mu, alpha, beta, tMax[0], processCount, maxSteps),
					2e-5,
				)
			}
		})
	})
}

func TestTensorBackend_applyHawkesGraph(test *testing.T) {
	Convey("Given Metal Hawkes graph execution", test, func() {
		Convey("It should execute Hawkes operations without intermediate readback", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, targets, expectedBytes, expected := hawkesGraph(test)

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
			assertHawkesGraphOutputs(results, expected)
		})
	})
}

func BenchmarkMetalHawkes_IntensityTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, hawkesOps := hawkesBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	processCount, eventCount := 8, 8192
	times := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, eventCount), hawkesTimes(eventCount))
	alpha := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, processCount), hawkesAlpha(processCount))
	beta := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, processCount), hawkesBeta(processCount))
	mu := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, processCount), hawkesMu(processCount))
	currentTime := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, 1), []float64{820})
	outputShape := hawkesBenchmarkShape(benchmark, processCount)
	defer closeBenchmarkTensors(times, alpha, beta, mu, currentTime)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := hawkesOps.IntensityTensor(times, alpha, beta, mu, currentTime, outputShape)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkMetalHawkes_KernelMatrixTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, hawkesOps := hawkesBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	eventCount := 128
	times := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, eventCount), hawkesTimes(eventCount))
	alpha := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, 1), []float64{0.7})
	beta := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, 1), []float64{0.4})
	outputShape := hawkesBenchmarkShape(benchmark, eventCount*eventCount)
	defer closeBenchmarkTensors(times, alpha, beta)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := hawkesOps.KernelMatrixTensor(times, alpha, beta, outputShape)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkMetalHawkes_LogLikelihoodTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, hawkesOps := hawkesBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	eventCount := 8192
	intensities := uploadMetalTensor(
		tensorBackend,
		hawkesBenchmarkShape(benchmark, eventCount),
		hawkesIntensities(eventCount),
	)
	integral := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, 1), []float64{1.25})
	outputShape := hawkesBenchmarkShape(benchmark, 1)
	defer closeBenchmarkTensors(intensities, integral)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := hawkesOps.LogLikelihoodTensor(intensities, integral, outputShape)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkMetalHawkes_SimulateTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, hawkesOps := hawkesBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	processCount, maxSteps := 8, 256
	mu := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, processCount), hawkesMu(processCount))
	alpha := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, processCount), hawkesZeros(processCount))
	beta := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, processCount), hawkesBeta(processCount))
	tMax := uploadMetalTensor(tensorBackend, hawkesBenchmarkShape(benchmark, 1), []float64{2})
	outputShape := hawkesBenchmarkShape(benchmark, processCount*maxSteps)
	defer closeBenchmarkTensors(mu, alpha, beta, tMax)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := hawkesOps.SimulateTensor(mu, alpha, beta, tMax, outputShape)
		closeBenchmarkOutput(benchmark, output, err)
	}
}
