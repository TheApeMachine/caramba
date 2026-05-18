package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkKernel_RunMathDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalMathDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkMathDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkMathDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("inv_sqrt_dim_scale", func(benchmark *testing.B) {
		benchmarkInvSqrtDimScale(benchmark, backend, storageDType)
	})
	benchmark.Run("logsumexp", func(benchmark *testing.B) {
		benchmarkLogSumExp(benchmark, backend, storageDType)
	})
	benchmark.Run("outer", func(benchmark *testing.B) {
		benchmarkOuter(benchmark, backend, storageDType)
	})
}

func benchmarkInvSqrtDimScale(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	elementCount := 8192
	input, dim, out := invSqrtDimScaleBenchmarkTensors(benchmark, backend, storageDType, elementCount)
	defer closeBenchmarkTensors(input, dim, out)

	benchmark.SetBytes(mathUnaryBenchmarkBytes(elementCount, storageDType) + 4)
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupInvSqrtDimScaleKernel(benchmark, storageDType).Run(input, dim, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkLogSumExp(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	rows, cols := 128, 1024
	input, out := logSumExpBenchmarkTensors(benchmark, backend, storageDType, rows, cols)
	defer closeBenchmarkTensors(input, out)

	benchmark.SetBytes(mathLogSumExpBenchmarkBytes(rows, cols, storageDType))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupLogSumExpKernel(benchmark, storageDType).Run(input, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkOuter(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	rows, cols := 128, 512
	left, right, out := outerBenchmarkTensors(benchmark, backend, storageDType, rows, cols)
	defer closeBenchmarkTensors(left, right, out)

	benchmark.SetBytes(mathOuterBenchmarkBytes(rows, cols, storageDType))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupOuterKernel(benchmark, storageDType).Run(left, right, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func invSqrtDimScaleBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	scaleDim := int32(17)
	fixture := invSqrtDimScaleFixtureForTest(elementCount, storageDType, scaleDim)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.inputBytes)
	dim := uploadInt32ScalarForTest(testingObject, backend, scaleDim)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)

	return input, dim, out
}

func logSumExpBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	rows int,
	cols int,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	fixture := logSumExpFixtureForTest(rows, cols, storageDType)
	inputShape := mustShapeForTest(testingObject, []int{rows, cols})
	outShape := mustShapeForTest(testingObject, []int{rows})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, out
}

func outerBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	rows int,
	cols int,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	fixture := outerFixtureForTest(rows, cols, storageDType)
	leftShape := mustShapeForTest(testingObject, []int{rows})
	rightShape := mustShapeForTest(testingObject, []int{cols})
	outShape := mustShapeForTest(testingObject, []int{rows, cols})
	left := uploadDTypeTensorForTest(testingObject, backend, leftShape, storageDType, fixture.leftBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, rightShape, storageDType, fixture.rightBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return left, right, out
}

func mathUnaryBenchmarkBytes(elementCount int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	return int64(elementCount) * elementBytes * 2
}

func mathLogSumExpBenchmarkBytes(rows int, cols int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	return int64(rows*cols+rows) * elementBytes
}

func mathOuterBenchmarkBytes(rows int, cols int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	return int64(rows+cols+rows*cols) * elementBytes
}
