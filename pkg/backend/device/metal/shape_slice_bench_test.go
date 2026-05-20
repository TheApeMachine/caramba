package metal

import (
	"context"
	"testing"
)

func BenchmarkKernel_RunSliceDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	seqLen := parityElementCounts[len(parityElementCounts)-1]

	for _, storageDType := range metalShapeDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			fixture := sliceFixtureForTest(seqLen, storageDType)
			inputShape := mustShapeForTest(benchmark, sliceInputShapeForTest(seqLen))
			outShape := mustShapeForTest(benchmark, sliceOutputShapeForTest(seqLen))
			input := uploadDTypeTensorForTest(benchmark, backend, inputShape, storageDType, fixture.inputBytes)
			dim := uploadInt32ScalarForTest(benchmark, backend, 1)
			start := uploadInt32ScalarForTest(benchmark, backend, 0)
			end := uploadInt32ScalarForTest(benchmark, backend, int32(seqLen))
			out := emptyTensorForTest(benchmark, backend, outShape, storageDType)
			defer closeBenchmarkTensors(input, dim, start, end, out)

			kernel := lookupSliceKernel(benchmark, storageDType)
			byteCount, err := outShape.Bytes(storageDType)
			if err != nil {
				benchmark.Fatal(err)
			}

			benchmark.SetBytes(int64(byteCount))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := kernel.Run(input, dim, start, end, out); err != nil {
					benchmark.Fatal(err)
				}

				if err := out.Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}
