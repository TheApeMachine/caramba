package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkKernel_RunExtendedUnaryElementwiseDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalExtendedUnaryDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkExtendedUnaryElementwiseDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkExtendedUnaryElementwiseDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, testCase := range extendedUnaryCases {
		testCase := testCase

		benchmark.Run(testCase.name, func(benchmark *testing.B) {
			shape, input, out := benchmarkExtendedUnaryTensors(
				benchmark,
				backend,
				testCase.name,
				storageDType,
			)
			defer closeBenchmarkTensors(input, out)

			elementBytes := dtypeBytesForBenchmark(storageDType)
			benchmark.SetBytes(int64(shape.Len() * 2 * elementBytes))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				err := runMetalUnaryElementwise(testCase.operation, input, out)
				if err != nil {
					benchmark.Fatal(err)
				}

				if err := out.Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func benchmarkExtendedUnaryTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) (tensor.Shape, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape, err := tensor.NewShape([]int{8192})
	if err != nil {
		testingObject.Fatal(err)
	}

	inputBytes, _ := extendedUnaryBytes(shape.Len(), name, storageDType)
	input := uploadTensorBytesForTest(testingObject, backend, shape, storageDType, inputBytes)

	out, err := backend.bridge.empty(shape, storageDType)
	if err != nil {
		closeBenchmarkTensors(input)
		testingObject.Fatal(err)
	}

	return shape, input, out
}
