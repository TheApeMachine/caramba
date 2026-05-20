package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunElementwiseDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range elementwiseStorageDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkBinaryElementwiseDType(benchmark, backend, storageDType)
			benchmarkUnaryElementwiseDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkBinaryElementwiseDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, testCase := range binaryFloat32Cases {
		testCase := testCase

		benchmark.Run(testCase.name, func(benchmark *testing.B) {
			shape, left, right, out := benchmarkBinaryElementwiseDTypeTensors(
				benchmark,
				backend,
				testCase.name,
				storageDType,
			)
			defer closeBenchmarkTensors(left, right, out)

			benchmark.SetBytes(int64(shape.Len() * 3 * dtypeBytesForBenchmark(storageDType)))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := runMetalBinaryElementwise(testCase.operation, left, right, out); err != nil {
					benchmark.Fatal(err)
				}

				if err := out.Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func benchmarkUnaryElementwiseDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, testCase := range unaryFloat32Cases {
		testCase := testCase

		benchmark.Run(testCase.name, func(benchmark *testing.B) {
			shape, input, out := benchmarkUnaryElementwiseDTypeTensors(
				benchmark,
				backend,
				testCase.name,
				storageDType,
			)
			defer closeBenchmarkTensors(input, out)

			benchmark.SetBytes(int64(shape.Len() * 2 * dtypeBytesForBenchmark(storageDType)))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := runMetalUnaryElementwise(testCase.operation, input, out); err != nil {
					benchmark.Fatal(err)
				}

				if err := out.Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func benchmarkBinaryElementwiseDTypeTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) (tensor.Shape, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape, err := tensor.NewShape([]int{8192})
	if err != nil {
		testingObject.Fatal(err)
	}

	leftBytes, rightBytes, _ := binaryElementwiseDTypeBytes(shape.Len(), name, storageDType)
	left := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, leftBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, rightBytes)

	out, err := backend.bridge.empty(shape, storageDType)
	if err != nil {
		closeBenchmarkTensors(left, right)
		testingObject.Fatal(err)
	}

	return shape, left, right, out
}

func benchmarkUnaryElementwiseDTypeTensors(
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

	inputBytes, _ := unaryElementwiseDTypeBytes(shape.Len(), name, storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, inputBytes)

	out, err := backend.bridge.empty(shape, storageDType)
	if err != nil {
		closeBenchmarkTensors(input)
		testingObject.Fatal(err)
	}

	return shape, input, out
}

func closeBenchmarkTensors(targets ...tensor.Tensor) {
	for _, target := range targets {
		_ = target.Close()
	}
}

func dtypeBytesForBenchmark(storageDType dtype.DType) int {
	size, err := storageDType.Size()
	if err != nil {
		panic(err)
	}

	return size
}
