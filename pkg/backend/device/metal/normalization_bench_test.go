package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunNormalizationDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalNormalizationDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkLayerNormDType(benchmark, backend, storageDType)
			benchmarkRMSNormDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkLayerNormDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("layernorm", func(benchmark *testing.B) {
		rows, cols := 128, 1024
		input, scale, bias, out := benchmarkLayerNormTensors(
			benchmark, backend, rows, cols, storageDType,
		)
		defer closeBenchmarkTensors(input, scale, bias, out)

		benchmark.SetBytes(normalizationBenchmarkBytes(rows, cols, storageDType, true))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalLayerNorm(input, scale, bias, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkRMSNormDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("rmsnorm", func(benchmark *testing.B) {
		rows, cols := 128, 1024
		input, scale, out := benchmarkRMSNormTensors(
			benchmark, backend, rows, cols, storageDType,
		)
		defer closeBenchmarkTensors(input, scale, out)

		benchmark.SetBytes(normalizationBenchmarkBytes(rows, cols, storageDType, false))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalRMSNorm(input, scale, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkLayerNormTensors(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	cols int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, scaleBytes, biasBytes := normDTypeBytes(rows, cols, storageDType)
	return layerNormTensorsForTest(
		testingObject, backend, rows, cols, storageDType, inputBytes, scaleBytes, biasBytes,
	)
}

func benchmarkRMSNormTensors(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	cols int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, scaleBytes, _ := normDTypeBytes(rows, cols, storageDType)
	return rmsNormTensorsForTest(testingObject, backend, rows, cols, storageDType, inputBytes, scaleBytes)
}

func normalizationBenchmarkBytes(
	rows int,
	cols int,
	storageDType dtype.DType,
	withBias bool,
) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	elements := rows*cols*2 + cols

	if withBias {
		elements += cols
	}

	return int64(elements) * elementBytes
}
