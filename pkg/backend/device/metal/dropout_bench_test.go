package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunDropoutDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalDropoutDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkDropoutDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkDropoutDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	elementCount := 8192
	input, out := dropoutBenchmarkTensors(benchmark, backend, storageDType, elementCount)
	defer closeBenchmarkTensors(input, out)

	benchmark.SetBytes(int64(elementCount * dtypeBytesForBenchmark(storageDType) * 2))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupDropoutKernel(benchmark, storageDType).Run(input, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func dropoutBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	fixture := dropoutFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)

	return input, out
}
