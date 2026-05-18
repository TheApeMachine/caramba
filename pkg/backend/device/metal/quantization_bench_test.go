package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func BenchmarkKernel_RunQuantization(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	benchmarkInt8Dequant(benchmark, backend)
	benchmarkInt4Dequant(benchmark, backend)
	benchmarkInt8Quant(benchmark, backend)
}

func benchmarkInt8Dequant(benchmark *testing.B, backend *Backend) {
	benchmark.Run("int8_dequant", func(benchmark *testing.B) {
		elementCount := 8192
		inputBytes, _ := int8DequantBytes(elementCount)
		input, out := quantizationBenchmarkTensors(
			benchmark, backend, elementCount, dtype.Int8, dtype.Float32, inputBytes,
		)
		defer closeBenchmarkTensors(input, out)

		benchmark.SetBytes(int64(elementCount * 5))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := lookupQuantizationKernel(benchmark, "int8_dequant", dtype.Int8, dtype.Float32).Run(input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkInt4Dequant(benchmark *testing.B, backend *Backend) {
	benchmark.Run("int4_dequant", func(benchmark *testing.B) {
		elementCount := 8192
		inputBytes, _ := int4DequantBytes(elementCount)
		input, out := quantizationBenchmarkTensors(
			benchmark, backend, elementCount, dtype.Int4, dtype.Float32, inputBytes,
		)
		defer closeBenchmarkTensors(input, out)

		benchmark.SetBytes(int64((elementCount+1)/2 + elementCount*4))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := lookupQuantizationKernel(benchmark, "int4_dequant", dtype.Int4, dtype.Float32).Run(input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkInt8Quant(benchmark *testing.B, backend *Backend) {
	benchmark.Run("int8_quant", func(benchmark *testing.B) {
		elementCount := 8192
		inputValues, _ := int8QuantValues(elementCount)
		input, out := quantizationBenchmarkTensors(
			benchmark, backend, elementCount, dtype.Float32, dtype.Int8,
			dtypeconvert.Float32ToBytes(inputValues),
		)
		defer closeBenchmarkTensors(input, out)

		benchmark.SetBytes(int64(elementCount * 5))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := lookupQuantizationKernel(benchmark, "int8_quant", dtype.Float32, dtype.Int8).Run(input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func quantizationBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
	inputDType dtype.DType,
	outDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor) {
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, inputDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, outDType)

	return input, out
}
