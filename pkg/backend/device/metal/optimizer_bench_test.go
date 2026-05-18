package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkKernel_RunOptimizerDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalOptimizerDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			for _, testCase := range optimizer4Cases {
				benchmarkOptimizer4DType(benchmark, backend, storageDType, testCase)
			}

			for _, testCase := range optimizer3Cases {
				benchmarkOptimizer3DType(benchmark, backend, storageDType, testCase)
			}

			benchmarkOptimizer2DType(benchmark, backend, storageDType)
			benchmarkHebbianDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkOptimizer4DType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
	testCase optimizer4Case,
) {
	benchmark.Run(testCase.name, func(benchmark *testing.B) {
		elementCount := 8192
		tensors := optimizer4BenchmarkTensors(benchmark, backend, storageDType, elementCount, testCase)
		defer closeBenchmarkTensors(tensors...)

		benchmark.SetBytes(optimizer4BenchmarkBytes(benchmark, elementCount, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := lookupOptimizer4Kernel(benchmark, testCase.name, storageDType).Run(tensors...); err != nil {
				benchmark.Fatal(err)
			}

			if err := tensors[4].Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkOptimizer3DType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
	testCase optimizer3Case,
) {
	benchmark.Run(testCase.name, func(benchmark *testing.B) {
		elementCount := 8192
		tensors := optimizer3BenchmarkTensors(benchmark, backend, storageDType, elementCount)
		defer closeBenchmarkTensors(tensors...)

		benchmark.SetBytes(optimizer3BenchmarkBytes(benchmark, elementCount, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := lookupOptimizer3Kernel(benchmark, testCase.name, storageDType).Run(tensors...); err != nil {
				benchmark.Fatal(err)
			}

			if err := tensors[3].Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkOptimizer2DType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("lbfgs_step", func(benchmark *testing.B) {
		elementCount := 8192
		tensors := optimizer2BenchmarkTensors(benchmark, backend, storageDType, elementCount)
		defer closeBenchmarkTensors(tensors...)

		benchmark.SetBytes(optimizer2BenchmarkBytes(benchmark, elementCount, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := lookupOptimizer2Kernel(benchmark, "lbfgs_step", storageDType).Run(tensors...); err != nil {
				benchmark.Fatal(err)
			}

			if err := tensors[2].Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkHebbianDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("hebbian_step", func(benchmark *testing.B) {
		elementCount := 8192
		weightBytes, postBytes, preBytes, _ := hebbianDTypeBytes(elementCount, storageDType)
		weights, post, pre, out := hebbianTensorsForTest(
			benchmark, backend, storageDType, elementCount, weightBytes, postBytes, preBytes,
		)
		defer closeBenchmarkTensors(weights, post, pre, out)

		benchmark.SetBytes(optimizerHebbianBenchmarkBytes(benchmark, elementCount, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := lookupHebbianKernel(benchmark, storageDType).Run(weights, post, pre, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func optimizer4BenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	testCase optimizer4Case,
) []tensor.Tensor {
	paramBytes, gradientBytes, _, _ := optimizerStorageInputs(elementCount, storageDType)
	firstState := optimizerStateValues(elementCount, 3)
	secondState := optimizerStateValues(elementCount, 5)
	if testCase.name == "adamax_step" {
		secondState = optimizerAdamaxInfinityValues(elementCount)
	}

	return optimizer4TensorsForTest(
		testingObject, backend, storageDType, elementCount,
		paramBytes, gradientBytes, firstState, secondState,
	)
}
