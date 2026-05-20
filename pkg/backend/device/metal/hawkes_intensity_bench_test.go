package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func BenchmarkKernel_RunHawkesIntensityDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	elementCount := parityElementCounts[len(parityElementCounts)-1]

	for _, storageDType := range metalHawkesMarkovDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			fixture := hawkesIntensityDTypeBytes(elementCount, storageDType)
			eventShape := mustShapeForTest(benchmark, []int{elementCount})
			outShape := mustShapeForTest(benchmark, []int{elementCount})
			events, queryTimes, baseline, alpha, beta, out := hawkesIntensityTensorsForTest(
				benchmark, backend, storageDType, eventShape, outShape, fixture,
			)
			defer closeBenchmarkTensors(events, queryTimes, baseline, alpha, beta, out)

			benchmark.SetBytes(hawkesIntensityBenchmarkBytes(elementCount, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := lookupHawkesIntensityKernel(benchmark, storageDType).Run(
					events, queryTimes, baseline, alpha, beta, out,
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

func hawkesIntensityBenchmarkBytes(elementCount int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))

	return int64(elementCount)*elementBytes*2 + elementBytes*3 + int64(elementCount)*elementBytes
}
