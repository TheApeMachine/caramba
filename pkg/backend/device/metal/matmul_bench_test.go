package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunMatMulDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalMatMulDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkMatMulDType(benchmark, backend, storageDType)
			benchmarkMatMulAddDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkMatMulDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("matmul", func(benchmark *testing.B) {
		rows, inner, cols := 128, 128, 128
		left, right, out := benchmarkMatMulTensors(
			benchmark, backend, rows, inner, cols, storageDType,
		)
		defer closeBenchmarkTensors(left, right, out)

		benchmark.SetBytes(matMulBenchmarkBytes(rows, inner, cols, storageDType, false))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalMatMul(left, right, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkMatMulAddDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("matmul_add", func(benchmark *testing.B) {
		rows, inner, cols := 128, 128, 128
		left, right, bias, out := benchmarkMatMulAddTensors(
			benchmark, backend, rows, inner, cols, storageDType,
		)
		defer closeBenchmarkTensors(left, right, bias, out)

		benchmark.SetBytes(matMulBenchmarkBytes(rows, inner, cols, storageDType, true))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalMatMulAdd(left, right, bias, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkMatMulTensors(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	inner int,
	cols int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	leftBytes, rightBytes, _, _ := matMulDTypeBytes(rows, inner, cols, storageDType, false)
	return matMulTensorsForTest(
		testingObject, backend, rows, inner, cols, storageDType, leftBytes, rightBytes,
	)
}

func benchmarkMatMulAddTensors(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	inner int,
	cols int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	leftBytes, rightBytes, biasBytes, _ := matMulDTypeBytes(rows, inner, cols, storageDType, true)
	left, right, out := matMulTensorsForTest(
		testingObject, backend, rows, inner, cols, storageDType, leftBytes, rightBytes,
	)
	bias := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{cols}), storageDType, biasBytes,
	)

	return left, right, bias, out
}

func matMulBenchmarkBytes(
	rows int,
	inner int,
	cols int,
	storageDType dtype.DType,
	withBias bool,
) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	elements := rows*inner + inner*cols + rows*cols

	if withBias {
		elements += cols
	}

	return int64(elements) * elementBytes
}
