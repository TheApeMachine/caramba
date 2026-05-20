package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func BenchmarkKernel_RunHawkesKernelMatrixDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	elementCount := parityElementCounts[len(parityElementCounts)-1]

	for _, storageDType := range metalHawkesMarkovDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			eventCount := hawkesMatrixEventCount(elementCount)
			fixture := hawkesKernelMatrixDTypeBytes(elementCount, storageDType)
			eventShape := mustShapeForTest(benchmark, []int{eventCount})
			outShape := mustShapeForTest(benchmark, []int{eventCount, eventCount})
			events, alpha, beta, out := hawkesKernelMatrixTensorsForTest(
				benchmark, backend, storageDType, eventShape, outShape, fixture,
			)
			defer closeBenchmarkTensors(events, alpha, beta, out)

			benchmark.SetBytes(hawkesKernelMatrixBenchmarkBytes(eventCount, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := lookupHawkesKernelMatrixKernel(benchmark, storageDType).Run(
					events, alpha, beta, out,
				); err != nil {
					benchmark.Fatal(err)
				}

				if err := out.Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func hawkesKernelMatrixBenchmarkBytes(eventCount int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	matrixBytes := int64(eventCount * eventCount * int(elementBytes))

	return int64(eventCount)*elementBytes + elementBytes*2 + matrixBytes
}
