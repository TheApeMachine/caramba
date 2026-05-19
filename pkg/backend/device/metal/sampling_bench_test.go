package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkKernel_RunSamplingDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalSamplingDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkSamplingDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkSamplingDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, name := range []string{"greedy_sample", "topk_sample", "topp_sample"} {
		name := name

		benchmark.Run(name, func(benchmark *testing.B) {
			benchmarkSamplingKernel(benchmark, backend, storageDType, name)
		})
	}
}

func benchmarkSamplingKernel(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
	name string,
) {
	elementCount := 8192
	input, out := samplingBenchmarkTensors(benchmark, backend, storageDType, name, elementCount)
	defer closeBenchmarkTensors(input, out)

	benchmark.SetBytes(int64(elementCount * dtypeBytesForBenchmark(storageDType)))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupSamplingKernel(benchmark, name, storageDType).Run(input, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func samplingBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	name string,
	elementCount int,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{1})
	inputBytes := samplingBenchmarkBytes(name, elementCount, storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, dtype.Int32)

	return input, out
}

func samplingBenchmarkBytes(name string, elementCount int, storageDType dtype.DType) []byte {
	if name == "greedy_sample" {
		return greedySamplingFixtureForTest(elementCount, storageDType).inputBytes
	}

	return drawSamplingFixtureForTest(elementCount, storageDType).inputBytes
}
