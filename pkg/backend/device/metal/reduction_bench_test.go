package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunReductionDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalReductionDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkReductionDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkReductionDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, name := range metalReductionNames {
		name := name

		benchmark.Run(name, func(benchmark *testing.B) {
			benchmarkReduction(benchmark, backend, name, storageDType)
		})
	}
}

func benchmarkReduction(
	benchmark *testing.B,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) {
	elementCount := 8192
	input, out := reductionBenchmarkTensors(benchmark, backend, name, storageDType, elementCount)
	defer closeBenchmarkTensors(input, out)

	benchmark.SetBytes(reductionBenchmarkBytes(elementCount, storageDType))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupReductionKernel(benchmark, name, storageDType).Run(input, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func reductionBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
	elementCount int,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	fixture := reductionFixtureForTest(name, elementCount, storageDType)
	inputShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{1})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, out
}

func reductionBenchmarkBytes(elementCount int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	return int64(elementCount)*elementBytes + elementBytes
}
