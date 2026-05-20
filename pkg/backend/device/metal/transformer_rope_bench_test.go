package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func benchmarkRoPEDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("rope", func(benchmark *testing.B) {
		seqLen, numHeads, headDim := 1024, 32, 128
		input, out := benchmarkRoPETensors(
			benchmark, backend, seqLen, numHeads, headDim, storageDType,
		)
		defer closeBenchmarkTensors(input, out)

		elementCount := seqLen * numHeads * headDim
		benchmark.SetBytes(int64(elementCount * dtypeBytesForBenchmark(storageDType) * 2))
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
		testingObject, backend, seqLen, numHeads, headDim, storageDType, fixture,
	)
}
