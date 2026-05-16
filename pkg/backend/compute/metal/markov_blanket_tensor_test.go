//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalMarkovBlanket_PartitionTensor(test *testing.T) {
	Convey("Given resident Metal Markov partition inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		markovOps := markovOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar partition reference", func() {
			for _, elementCount := range metalContractSizes() {
				state, masks, packedMasks, counts := markovPartitionInputs(elementCount)
				outputShape := markovShape(test, counts[0]+counts[1]+counts[2]+counts[3])
				output, err := markovOps.PartitionTensor(
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, elementCount), state),
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, len(packedMasks)), packedMasks),
					outputShape,
					counts[0],
					counts[1],
					counts[2],
					counts[3],
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceMarkovPartition(state, masks, counts), 1e-6)
			}
		})
	})
}

func TestMetalMarkovBlanket_FlowInternalTensor(test *testing.T) {
	Convey("Given resident Metal Markov internal-flow inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		markovOps := markovOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar internal-flow reference", func() {
			for _, sensoryCount := range metalContractSizes() {
				internalCount := 3
				sensory := markovVector(sensoryCount, 0.012, -0.1)
				weights := markovWeights(internalCount, sensoryCount)
				bias := markovVector(internalCount, 0.004, 0.01)
				output, err := markovOps.FlowInternalTensor(
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, sensoryCount), sensory),
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, len(weights)), weights),
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, internalCount), bias),
					markovShape(test, internalCount),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceMarkovFlow(sensory, weights, bias, internalCount, sensoryCount), 1e-5)
			}
		})
	})
}

func TestMetalMarkovBlanket_FlowActiveTensor(test *testing.T) {
	Convey("Given resident Metal Markov active-flow inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		markovOps := markovOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar active-flow reference", func() {
			for _, internalCount := range metalContractSizes() {
				activeCount := 3
				internal := markovVector(internalCount, 0.018, 0.03)
				weights := markovWeights(activeCount, internalCount)
				bias := markovVector(activeCount, 0.005, -0.02)
				output, err := markovOps.FlowActiveTensor(
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, internalCount), internal),
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, len(weights)), weights),
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, activeCount), bias),
					markovShape(test, activeCount),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceMarkovFlow(internal, weights, bias, activeCount, internalCount), 1e-5)
			}
		})
	})
}

func TestMetalMarkovBlanket_MutualInformationTensor(test *testing.T) {
	Convey("Given resident Metal Markov mutual-information inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		markovOps := markovOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar Gaussian estimate", func() {
			for _, samples := range []int{7, 64, 1024, 8192} {
				xDimensions, yDimensions := 1, 1
				xValues, yValues := markovMutualInformationInputs(samples, xDimensions, yDimensions)
				output, err := markovOps.MutualInformationTensor(
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, len(xValues)), xValues),
					uploadMetalTensorForTest(test, tensorBackend, markovShape(test, len(yValues)), yValues),
					markovShape(test, 1),
					xDimensions,
					yDimensions,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				expected := referenceMarkovMutualInformation(xValues, yValues, samples, xDimensions, yDimensions)
				assertMetalMaxDiff(values, []float64{expected}, 2e-4)
			}
		})
	})
}

func TestTensorBackend_applyMarkovBlanketGraph(test *testing.T) {
	Convey("Given Metal Markov blanket graph execution", test, func() {
		Convey("It should keep Markov blanket graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, targets, expectedBytes, expected, tolerances := markovGraph(test)

			before := tensorBackend.runtime.Metrics()
			results, err := NewRunnerWithBackend(tensorBackend).Execute(context.Background(), graph, targets)
			after := tensorBackend.runtime.Metrics()
			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, len(targets))
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, expectedBytes)
			assertMarkovGraphOutputs(results, expected, tolerances)
		})
	})
}

func BenchmarkMetalMarkovBlanket_PartitionTensor(benchmark *testing.B) {
	benchmarkMarkov(benchmark, "partition")
}

func BenchmarkMetalMarkovBlanket_FlowInternalTensor(benchmark *testing.B) {
	benchmarkMarkov(benchmark, "flow_internal")
}

func BenchmarkMetalMarkovBlanket_FlowActiveTensor(benchmark *testing.B) {
	benchmarkMarkov(benchmark, "flow_active")
}

func BenchmarkMetalMarkovBlanket_MutualInformationTensor(benchmark *testing.B) {
	benchmarkMarkov(benchmark, "mutual_information")
}

func benchmarkMarkov(benchmark *testing.B, operation string) {
	benchmark.ReportAllocs()
	tensorBackend, markovOps := markovBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	markovBenchmarkLoop(benchmark, tensorBackend, markovOps, operation)
}

func markovBenchmarkLoop(
	benchmark *testing.B,
	tensorBackend *TensorBackend,
	markovOps *MetalMarkovBlanket,
	operation string,
) {
	state, _, packedMasks, counts := markovPartitionInputs(8192)
	sensoryCount, internalCount, activeCount := 8192, 8, 8
	sensory := uploadMetalTensor(tensorBackend, markovShape(benchmark, sensoryCount), markovVector(sensoryCount, 0.012, -0.1))
	internalWeights := uploadMetalTensor(tensorBackend, markovShape(benchmark, internalCount*sensoryCount), markovWeights(internalCount, sensoryCount))
	internalBias := uploadMetalTensor(tensorBackend, markovShape(benchmark, internalCount), markovVector(internalCount, 0.004, 0.01))
	internal := uploadMetalTensor(tensorBackend, markovShape(benchmark, sensoryCount), markovVector(sensoryCount, 0.018, 0.03))
	activeWeights := uploadMetalTensor(tensorBackend, markovShape(benchmark, activeCount*sensoryCount), markovWeights(activeCount, sensoryCount))
	activeBias := uploadMetalTensor(tensorBackend, markovShape(benchmark, activeCount), markovVector(activeCount, 0.005, -0.02))
	partitionState := uploadMetalTensor(tensorBackend, markovShape(benchmark, len(state)), state)
	partitionMasks := uploadMetalTensor(tensorBackend, markovShape(benchmark, len(packedMasks)), packedMasks)
	xValues, yValues := markovMutualInformationInputs(256, 4, 4)
	miX := uploadMetalTensor(tensorBackend, markovShape(benchmark, len(xValues)), xValues)
	miY := uploadMetalTensor(tensorBackend, markovShape(benchmark, len(yValues)), yValues)
	defer closeBenchmarkTensors(
		sensory, internalWeights, internalBias, internal, activeWeights,
		activeBias, partitionState, partitionMasks, miX, miY,
	)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := markovBenchmarkOutput(
			benchmark, markovOps, operation, partitionState, partitionMasks, counts,
			sensory, internalWeights, internalBias, internal, activeWeights, activeBias, miX, miY,
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func markovBenchmarkOutput(
	benchmark *testing.B,
	markovOps *MetalMarkovBlanket,
	operation string,
	partitionState computetensor.Float64Tensor,
	partitionMasks computetensor.Float64Tensor,
	counts []int,
	sensory computetensor.Float64Tensor,
	internalWeights computetensor.Float64Tensor,
	internalBias computetensor.Float64Tensor,
	internal computetensor.Float64Tensor,
	activeWeights computetensor.Float64Tensor,
	activeBias computetensor.Float64Tensor,
	miX computetensor.Float64Tensor,
	miY computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	switch operation {
	case "partition":
		return markovOps.PartitionTensor(
			partitionState,
			partitionMasks,
			markovShape(benchmark, partitionState.Shape().Len()),
			counts[0],
			counts[1],
			counts[2],
			counts[3],
		)
	case "flow_internal":
		return markovOps.FlowInternalTensor(sensory, internalWeights, internalBias, markovShape(benchmark, internalBias.Shape().Len()))
	case "flow_active":
		return markovOps.FlowActiveTensor(internal, activeWeights, activeBias, markovShape(benchmark, activeBias.Shape().Len()))
	case "mutual_information":
		return markovOps.MutualInformationTensor(miX, miY, markovShape(benchmark, 1), 4, 4)
	default:
		benchmark.Fatalf("unknown Markov benchmark operation %q", operation)
		return nil, nil
	}
}
