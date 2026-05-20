package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunUtilityKernels(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	benchmark.Run("checkpoint_encode_float32", func(benchmark *testing.B) {
		benchmarkCheckpointEncode(benchmark, backend)
	})
	benchmark.Run("checkpoint_decode_float32", func(benchmark *testing.B) {
		benchmarkCheckpointDecode(benchmark, backend)
	})
	benchmark.Run("tokenizer_pack_int32", func(benchmark *testing.B) {
		benchmarkTokenizerPack(benchmark, backend)
	})

	for _, storageDType := range metalUtilityFloatDTypes {
		storageDType := storageDType

		benchmark.Run("weight_freeze_mask/"+storageDType.Name(), func(benchmark *testing.B) {
			benchmarkWeightFreezeMask(benchmark, backend, storageDType)
		})
	}
}

func benchmarkCheckpointEncode(benchmark *testing.B, backend *Backend) {
	input, out := checkpointEncodeBenchmarkTensors(benchmark, backend, 8192)
	defer closeBenchmarkTensors(input, out)

	benchmark.SetBytes(int64(8192 * 8))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupCheckpointEncodeKernel(benchmark).Run(input, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkCheckpointDecode(benchmark *testing.B, backend *Backend) {
	input, out := checkpointDecodeBenchmarkTensors(benchmark, backend, 8192)
	defer closeBenchmarkTensors(input, out)

	benchmark.SetBytes(int64(8192 * 8))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupCheckpointDecodeKernel(benchmark).Run(input, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkTokenizerPack(benchmark *testing.B, backend *Backend) {
	input, out := tokenizerPackBenchmarkTensors(benchmark, backend, 8192)
	defer closeBenchmarkTensors(input, out)

	benchmark.SetBytes(int64(8192 * 8))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupTokenizerPackKernel(benchmark).Run(input, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkWeightFreezeMask(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	inputs := weightFreezeBenchmarkTensors(benchmark, backend, storageDType, 8192)
	defer closeBenchmarkTensors(inputs...)

	benchmark.SetBytes(int64(8192 * dtypeBytesForBenchmark(storageDType) * 2))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupWeightFreezeMaskKernel(benchmark, storageDType).Run(inputs...); err != nil {
			benchmark.Fatal(err)
		}

		if err := inputs[2].Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func checkpointEncodeBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) (tensor.Tensor, tensor.Tensor) {
	fixture := checkpointFixtureForTest(elementCount)
	inputShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{len(fixture.encodedBytes)})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, dtype.Float32, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, dtype.Uint8)

	return input, out
}

func checkpointDecodeBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) (tensor.Tensor, tensor.Tensor) {
	fixture := checkpointFixtureForTest(elementCount)
	inputShape := mustShapeForTest(testingObject, []int{len(fixture.encodedBytes)})
	outShape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, dtype.Uint8, fixture.encodedBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, dtype.Float32)

	return input, out
}

func tokenizerPackBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) (tensor.Tensor, tensor.Tensor) {
	fixture := tokenizerFixtureForTest(elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Int32, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, dtype.Int32)

	return input, out
}

func weightFreezeBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) []tensor.Tensor {
	fixture := weightFreezeFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	mask := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Bool, fixture.maskBytes)
	gradients := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.gradientBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)

	return []tensor.Tensor{mask, gradients, out}
}
