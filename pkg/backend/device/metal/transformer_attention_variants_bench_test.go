package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func benchmarkAttentionVariantsDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, variant := range metalAttentionVariantCases() {
		variant := variant

		benchmark.Run(variant.name, func(benchmark *testing.B) {
			benchmarkAttentionVariantDType(benchmark, backend, storageDType, variant)
		})
	}
}

func benchmarkAttentionVariantDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
	variant metalAttentionVariantCase,
) {
	seqQ, seqK, numHeads, headDim := 64, 64, 8, 64
	query, key, value, out := benchmarkAttentionVariantTensors(
		benchmark, backend, seqQ, seqK, numHeads, headDim, storageDType, variant,
	)
	defer closeBenchmarkTensors(query, key, value, out)

	benchmark.SetBytes(attentionVariantBenchmarkBytes(seqQ, seqK, numHeads, headDim, variant, storageDType))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := runMetalAttentionVariantBenchmark(variant, query, key, value, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkAttentionVariantTensors(
	testingObject testing.TB,
	backend *Backend,
	seqQ int,
	seqK int,
	numHeads int,
	headDim int,
	storageDType dtype.DType,
	variant metalAttentionVariantCase,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	fixture := attentionVariantFixtureForTest(
		seqQ, seqK, numHeads, headDim, storageDType, variant,
	)
	return attentionVariantTensorsForTest(
		testingObject, backend, seqQ, seqK, numHeads, headDim,
		storageDType, variant, fixture,
	)
}

func attentionVariantBenchmarkBytes(
	seqQ int,
	seqK int,
	numHeads int,
	headDim int,
	variant metalAttentionVariantCase,
	storageDType dtype.DType,
) int64 {
	dtypeBytes := dtypeBytesForBenchmark(storageDType)
	queryBytes := seqQ * numHeads * headDim * dtypeBytes
	kvBytes := seqK * variant.kvHeads * headDim * dtypeBytes * 2
	outBytes := seqQ * numHeads * headDim * dtypeBytes
	return int64(queryBytes + kvBytes + outBytes)
}

func runMetalAttentionVariantBenchmark(
	variant metalAttentionVariantCase,
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
	out tensor.Tensor,
) error {
	switch variant.name {
	case "grouped_query_attention":
		return runMetalGroupedQueryAttention(query, key, value, out)
	case "sliding_window_attention":
		return runMetalSlidingWindowAttention(query, key, value, out)
	default:
		return runMetalMultiHeadAttention(query, key, value, out)
	}
}
