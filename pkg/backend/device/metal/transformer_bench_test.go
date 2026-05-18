package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkKernel_RunTransformerDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalTransformerDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkAttentionDType(benchmark, backend, storageDType)
			benchmarkRoPEDType(benchmark, backend, storageDType)
			benchmarkEmbeddingLookupDType(benchmark, backend, storageDType)
			benchmarkEmbeddingBagDType(benchmark, backend, storageDType)
			benchmarkApplyMaskDType(benchmark, backend, storageDType)
			benchmarkCausalMaskDType(benchmark, backend, storageDType)
			benchmarkALiBiBiasDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkAttentionDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("attention", func(benchmark *testing.B) {
		seqQ, seqK, depth, valueDim := 64, 64, 128, 64
		query, key, value, out := benchmarkAttentionTensors(
			benchmark, backend, seqQ, seqK, depth, valueDim, storageDType,
		)
		defer closeBenchmarkTensors(query, key, value, out)

		benchmark.SetBytes(attentionBenchmarkBytes(seqQ, seqK, depth, valueDim, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalAttention(query, key, value, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkRoPEDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("rope", func(benchmark *testing.B) {
		seqLen, numHeads, headDim := 512, 8, 64
		input, out := benchmarkRoPETensors(
			benchmark, backend, seqLen, numHeads, headDim, storageDType,
		)
		defer closeBenchmarkTensors(input, out)

		benchmark.SetBytes(int64(seqLen * numHeads * headDim * dtypeBytesForBenchmark(storageDType) * 2))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalRoPE(input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkEmbeddingLookupDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("embedding_lookup", func(benchmark *testing.B) {
		vocab, hidden, indexCount := 32768, 512, 256
		table, indices, out := benchmarkEmbeddingLookupTensors(
			benchmark, backend, vocab, hidden, indexCount, storageDType,
		)
		defer closeBenchmarkTensors(table, indices, out)

		benchmark.SetBytes(int64(indexCount * hidden * dtypeBytesForBenchmark(storageDType) * 2))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalEmbeddingLookup(table, indices, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkEmbeddingBagDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("embedding_bag", func(benchmark *testing.B) {
		vocab, hidden, indexCount, bagCount := 32768, 512, 512, 64
		table, indices, offsets, out := benchmarkEmbeddingBagTensors(
			benchmark, backend, vocab, hidden, indexCount, bagCount, storageDType,
		)
		defer closeBenchmarkTensors(table, indices, offsets, out)

		benchmark.SetBytes(int64((indexCount+bagCount)*hidden) * int64(dtypeBytesForBenchmark(storageDType)))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalEmbeddingBag(table, indices, offsets, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkApplyMaskDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("apply_mask", func(benchmark *testing.B) {
		elementCount := 262144
		input, mask, out := benchmarkApplyMaskTensors(benchmark, backend, elementCount, storageDType)
		defer closeBenchmarkTensors(input, mask, out)

		benchmark.SetBytes(int64(elementCount * dtypeBytesForBenchmark(storageDType) * 3))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalApplyMask(input, mask, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkCausalMaskDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("causal_mask", func(benchmark *testing.B) {
		rows, cols := 512, 512
		input, out := benchmarkCausalMaskTensors(benchmark, backend, rows, cols, storageDType)
		defer closeBenchmarkTensors(input, out)

		benchmark.SetBytes(int64(rows * cols * dtypeBytesForBenchmark(storageDType)))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalCausalMask(input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkALiBiBiasDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("alibi_bias", func(benchmark *testing.B) {
		rows, cols := 512, 512
		scores, slope, out := benchmarkALiBiBiasTensors(benchmark, backend, rows, cols, storageDType)
		defer closeBenchmarkTensors(scores, slope, out)

		benchmark.SetBytes(int64(rows * cols * dtypeBytesForBenchmark(storageDType) * 2))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalALiBiBias(scores, slope, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkAttentionTensors(
	testingObject testing.TB,
	backend *Backend,
	seqQ int,
	seqK int,
	depth int,
	valueDim int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	fixture := attentionFixtureForTest(seqQ, seqK, depth, valueDim, storageDType)
	return attentionTensorsForTest(
		testingObject, backend, seqQ, seqK, depth, valueDim, storageDType, fixture,
	)
}

func benchmarkRoPETensors(
	testingObject testing.TB,
	backend *Backend,
	seqLen int,
	numHeads int,
	headDim int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	fixture := ropeFixtureForTest(seqLen, numHeads, headDim, storageDType)
	return ropeTensorsForTest(
		testingObject, backend, seqLen, numHeads, headDim, storageDType, fixture.inputBytes,
	)
}

func benchmarkEmbeddingLookupTensors(
	testingObject testing.TB,
	backend *Backend,
	vocab int,
	hidden int,
	indexCount int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	tableBytes, indicesBytes, _ := embeddingLookupDTypeBytes(vocab, hidden, indexCount, storageDType)
	return embeddingLookupTensorsForTest(
		testingObject, backend, vocab, hidden, indexCount, storageDType, tableBytes, indicesBytes,
	)
}

func benchmarkEmbeddingBagTensors(
	testingObject testing.TB,
	backend *Backend,
	vocab int,
	hidden int,
	indexCount int,
	bagCount int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	tableBytes, indicesBytes, offsetsBytes, _ := embeddingBagDTypeBytes(
		vocab, hidden, indexCount, bagCount, storageDType,
	)
	return embeddingBagTensorsForTest(
		testingObject, backend, vocab, hidden, bagCount, storageDType,
		tableBytes, indicesBytes, offsetsBytes,
	)
}

func benchmarkApplyMaskTensors(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, maskBytes, _ := applyMaskDTypeBytes(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, inputBytes)
	mask := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, maskBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return input, mask, out
}

func benchmarkCausalMaskTensors(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	cols int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	input := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{1}),
		storageDType, encodeProjectionValuesAsDType([]float32{0}, storageDType),
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{rows, cols}),
		storageDType,
	)
	return input, out
}

func benchmarkALiBiBiasTensors(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	cols int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	scoresBytes, slopeBytes, _ := alibiBiasDTypeBytes(rows, cols, storageDType)
	shape := mustShapeForTest(testingObject, []int{rows, cols})
	scores := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, scoresBytes)
	slope := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{1}),
		storageDType, slopeBytes,
	)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return scores, slope, out
}

func attentionBenchmarkBytes(
	seqQ int,
	seqK int,
	depth int,
	valueDim int,
	storageDType dtype.DType,
) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	storedElements := seqQ*depth + seqK*depth + seqK*valueDim + seqQ*valueDim
	scoreBytes := int64(seqQ * seqK * 4)

	return int64(storedElements)*elementBytes + scoreBytes
}
