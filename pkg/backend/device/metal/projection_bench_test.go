package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunProjectionDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalProjectionDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkLinearDType(benchmark, backend, storageDType)
			benchmarkFusedQKVDType(benchmark, backend, storageDType)
			benchmarkLoRAMergeDType(benchmark, backend, storageDType)
			benchmarkLoRAApplyDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkLinearDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("linear", func(benchmark *testing.B) {
		batch, inner, outDim := 128, 512, 256
		input, weight, bias, out := benchmarkLinearTensors(
			benchmark, backend, batch, inner, outDim, storageDType,
		)
		defer closeBenchmarkTensors(input, weight, bias, out)

		benchmark.SetBytes(linearBenchmarkBytes(batch, inner, outDim, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalLinear(input, weight, bias, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkFusedQKVDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("fused_qkv", func(benchmark *testing.B) {
		batch, inner, outDim := 64, 512, 256
		input, weight, bias, query, key, value := benchmarkFusedQKVTensors(
			benchmark, backend, batch, inner, outDim, storageDType,
		)
		defer closeBenchmarkTensors(input, weight, bias, query, key, value)

		benchmark.SetBytes(fusedQKVBenchmarkBytes(batch, inner, outDim, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalFusedQKV(input, weight, bias, query, key, value); err != nil {
				benchmark.Fatal(err)
			}

			if err := query.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkLoRAMergeDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("lora_merge", func(benchmark *testing.B) {
		outDim, rank, inner := 512, 8, 512
		base, loraA, loraB, out := benchmarkLoRAMergeTensors(
			benchmark, backend, outDim, rank, inner, storageDType,
		)
		defer closeBenchmarkTensors(base, loraA, loraB, out)

		benchmark.SetBytes(loraMergeBenchmarkBytes(outDim, rank, inner, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalLoRAMerge(base, loraA, loraB, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkLoRAApplyDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("lora_apply", func(benchmark *testing.B) {
		batch, outDim, rank, inner := 64, 512, 8, 512
		base, loraA, loraB, input, out := benchmarkLoRAApplyTensors(
			benchmark, backend, batch, outDim, rank, inner, storageDType,
		)
		defer closeBenchmarkTensors(base, loraA, loraB, input, out)

		benchmark.SetBytes(loraApplyBenchmarkBytes(batch, outDim, rank, inner, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalLoRAApply(base, loraA, loraB, input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkLinearTensors(
	testingObject testing.TB,
	backend *Backend,
	batch int,
	inner int,
	outDim int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, weightBytes, biasBytes, _ := linearDTypeBytes(batch, inner, outDim, storageDType)
	return linearTensorsForTest(
		testingObject, backend, batch, inner, outDim, storageDType,
		inputBytes, weightBytes, biasBytes,
	)
}

func benchmarkFusedQKVTensors(
	testingObject testing.TB,
	backend *Backend,
	batch int,
	inner int,
	outDim int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, weightBytes, biasBytes, _ := fusedQKVDTypeBytes(
		batch, inner, outDim, storageDType,
	)
	return fusedQKVTensorsForTest(
		testingObject, backend, batch, inner, outDim, storageDType,
		inputBytes, weightBytes, biasBytes,
	)
}

func benchmarkLoRAMergeTensors(
	testingObject testing.TB,
	backend *Backend,
	outDim int,
	rank int,
	inner int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	baseBytes, loraABytes, loraBBytes, _ := loraMergeDTypeBytes(
		outDim, rank, inner, storageDType,
	)
	return loraMergeTensorsForTest(
		testingObject, backend, outDim, rank, inner, storageDType,
		baseBytes, loraABytes, loraBBytes,
	)
}

func benchmarkLoRAApplyTensors(
	testingObject testing.TB,
	backend *Backend,
	batch int,
	outDim int,
	rank int,
	inner int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	baseBytes, loraABytes, loraBBytes, inputBytes, _ := loraApplyDTypeBytes(
		batch, outDim, rank, inner, storageDType,
	)
	return loraApplyTensorsForTest(
		testingObject, backend, batch, outDim, rank, inner, storageDType,
		baseBytes, loraABytes, loraBBytes, inputBytes,
	)
}

func linearBenchmarkBytes(batch int, inner int, outDim int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	elements := batch*inner + outDim*inner + outDim + batch*outDim
	return int64(elements) * elementBytes
}

func fusedQKVBenchmarkBytes(batch int, inner int, outDim int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	elements := batch*inner + 3*outDim*inner + 3*outDim + 3*batch*outDim
	return int64(elements) * elementBytes
}

func loraMergeBenchmarkBytes(outDim int, rank int, inner int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	elements := outDim*inner + outDim*rank + rank*inner + outDim*inner
	return int64(elements) * elementBytes
}

func loraApplyBenchmarkBytes(
	batch int,
	outDim int,
	rank int,
	inner int,
	storageDType dtype.DType,
) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	elements := batch*outDim + outDim*rank + rank*inner + batch*inner + batch*outDim
	return int64(elements) * elementBytes
}
