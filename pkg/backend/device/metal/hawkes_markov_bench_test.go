package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunHawkesMarkovDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalHawkesMarkovDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkHawkesMarkovDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkHawkesMarkovDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, name := range []string{
		"hawkes_intensity",
		"hawkes_kernel_matrix",
		"hawkes_log_likelihood",
		"markov_mutual_information",
		"markov_blanket_partition",
		"markov_flow_active",
		"markov_flow_internal",
	} {
		name := name

		benchmark.Run(name, func(benchmark *testing.B) {
			tensors := benchmarkHawkesMarkovTensors(benchmark, backend, name, storageDType)
			defer closeBenchmarkTensors(tensors...)

			benchmark.SetBytes(hawkesMarkovBenchmarkBytes(name, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := benchmarkHawkesMarkovKernel(benchmark, name, storageDType).Run(tensors...); err != nil {
					benchmark.Fatal(err)
				}

				if err := tensors[len(tensors)-1].Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func benchmarkHawkesMarkovTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) []tensor.Tensor {
	switch name {
	case "hawkes_intensity":
		return benchmarkHawkesIntensityTensors(testingObject, backend, storageDType)
	case "hawkes_kernel_matrix":
		return benchmarkHawkesKernelMatrixTensors(testingObject, backend, storageDType)
	case "hawkes_log_likelihood":
		return benchmarkHawkesLogLikelihoodTensors(testingObject, backend, storageDType)
	case "markov_mutual_information":
		return benchmarkMarkovMutualInformationTensors(testingObject, backend, storageDType)
	case "markov_blanket_partition":
		return benchmarkMarkovPartitionTensors(testingObject, backend, storageDType)
	default:
		return benchmarkMarkovFlowTensors(testingObject, backend, name, storageDType)
	}
}

func benchmarkHawkesIntensityTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	eventCount, queryCount := 2048, 1024
	fixture := hawkesIntensityFixtureForTest(storageDType, eventCount, queryCount)
	eventShape := mustShapeForTest(testingObject, []int{eventCount})
	outShape := mustShapeForTest(testingObject, []int{queryCount})
	events, queries, baseline, alpha, beta, out := hawkesFiveTensorsForTest(
		testingObject, backend, storageDType, eventShape, outShape, fixture,
	)
	return []tensor.Tensor{events, queries, baseline, alpha, beta, out}
}

func benchmarkHawkesKernelMatrixTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	eventCount := 256
	fixture := hawkesKernelMatrixFixtureForTest(storageDType, eventCount)
	eventShape := mustShapeForTest(testingObject, []int{eventCount})
	outShape := mustShapeForTest(testingObject, []int{eventCount, eventCount})
	events := uploadDTypeTensorForTest(testingObject, backend, eventShape, storageDType, fixture.firstBytes)
	alpha := uploadDTypeTensorForTest(testingObject, backend, scalarShapeForTest(testingObject), storageDType, fixture.thirdBytes)
	beta := uploadDTypeTensorForTest(testingObject, backend, scalarShapeForTest(testingObject), storageDType, fixture.fourthBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	return []tensor.Tensor{events, alpha, beta, out}
}

func benchmarkHawkesLogLikelihoodTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	eventCount := 2048
	fixture := hawkesLogLikelihoodFixtureForTest(storageDType, eventCount)
	eventShape := mustShapeForTest(testingObject, []int{eventCount})
	outShape := scalarShapeForTest(testingObject)
	events, totalTime, baseline, alpha, beta, out := hawkesFiveTensorsForTest(
		testingObject, backend, storageDType, eventShape, outShape, fixture,
	)
	return []tensor.Tensor{events, totalTime, baseline, alpha, beta, out}
}

func benchmarkMarkovMutualInformationTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	rows, cols := 64, 64
	fixture := markovMutualInformationFixtureForTest(storageDType, rows, cols)
	inputShape := mustShapeForTest(testingObject, []int{rows, cols})
	outShape := scalarShapeForTest(testingObject)
	joint := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.firstBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	return []tensor.Tensor{joint, out}
}

func benchmarkMarkovPartitionTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	nodeCount := 256
	fixture := markovPartitionFixtureForTest(storageDType, nodeCount)
	matrixShape := mustShapeForTest(testingObject, []int{nodeCount, nodeCount})
	labelShape := mustShapeForTest(testingObject, []int{len(fixture.labels)})
	outShape := mustShapeForTest(testingObject, []int{nodeCount})
	adjacency := uploadDTypeTensorForTest(testingObject, backend, matrixShape, storageDType, fixture.firstBytes)
	internal := uploadDTypeTensorForTest(testingObject, backend, labelShape, dtype.Int32, int32ValuesToBytes(fixture.labels))
	out := emptyTensorForTest(testingObject, backend, outShape, dtype.Int32)
	return []tensor.Tensor{adjacency, internal, out}
}

func benchmarkMarkovFlowTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) []tensor.Tensor {
	nodeCount := 256
	fixture := markovFlowFixtureForTest(name, storageDType, nodeCount)
	matrixShape := mustShapeForTest(testingObject, []int{nodeCount, nodeCount})
	labelShape := mustShapeForTest(testingObject, []int{nodeCount})
	mi := uploadDTypeTensorForTest(testingObject, backend, matrixShape, storageDType, fixture.firstBytes)
	partition := uploadDTypeTensorForTest(testingObject, backend, labelShape, dtype.Int32, int32ValuesToBytes(fixture.labels))
	out := emptyTensorForTest(testingObject, backend, labelShape, storageDType)
	return []tensor.Tensor{mi, partition, out}
}

func benchmarkHawkesMarkovKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	switch name {
	case "hawkes_intensity":
		return lookupHawkesIntensityKernel(testingObject, storageDType)
	case "hawkes_kernel_matrix":
		return lookupHawkesKernelMatrixKernel(testingObject, storageDType)
	case "hawkes_log_likelihood":
		return lookupHawkesLogLikelihoodKernel(testingObject, storageDType)
	case "markov_mutual_information":
		return lookupMarkovMutualInformationKernel(testingObject, storageDType)
	case "markov_blanket_partition":
		return lookupMarkovBlanketPartitionKernel(testingObject, storageDType)
	default:
		return lookupMarkovFlowKernel(testingObject, name, storageDType)
	}
}

func hawkesMarkovBenchmarkBytes(name string, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))

	switch name {
	case "hawkes_intensity":
		return int64(2048+1024+4+1024) * elementBytes
	case "hawkes_kernel_matrix":
		return int64(256+2+256*256) * elementBytes
	case "hawkes_log_likelihood":
		return int64(2048+4+1) * elementBytes
	case "markov_mutual_information":
		return int64(64*64+1) * elementBytes
	case "markov_blanket_partition":
		return int64(256*256)*elementBytes + int64(258*4)
	default:
		return int64(256*256+256)*elementBytes + int64(256*4)
	}
}
