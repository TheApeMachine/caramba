package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunSoftmaxDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalSoftmaxDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkSoftmaxDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkSoftmaxDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	rows, cols := 128, 1024
	input, out := benchmarkSoftmaxTensors(benchmark, backend, rows, cols, storageDType)
	defer closeBenchmarkTensors(input, out)

	benchmark.SetBytes(softmaxBenchmarkBytes(rows, cols, storageDType))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := runMetalSoftmax(input, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkSoftmaxTensors(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	cols int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, _ := softmaxDTypeBytes(rows, cols, storageDType)
	return softmaxTensorsForTest(testingObject, backend, rows, cols, storageDType, inputBytes)
}

func softmaxBenchmarkBytes(rows int, cols int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	return int64(rows*cols) * elementBytes * 2
}
