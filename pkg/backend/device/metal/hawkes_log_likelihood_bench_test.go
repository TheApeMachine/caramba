package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func BenchmarkKernel_RunHawkesLogLikelihoodDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	elementCount := parityElementCounts[len(parityElementCounts)-1]

	for _, storageDType := range metalHawkesMarkovDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			fixture := hawkesLogLikelihoodDTypeBytes(elementCount, storageDType)
			eventShape := mustShapeForTest(benchmark, []int{elementCount})
			outShape := scalarShapeForTest(benchmark)
			events, totalTime, baseline, alpha, beta, out := hawkesLogLikelihoodTensorsForTest(
				benchmark, backend, storageDType, eventShape, outShape, fixture,
			)
			defer closeBenchmarkTensors(events, totalTime, baseline, alpha, beta, out)

			benchmark.SetBytes(hawkesLogLikelihoodBenchmarkBytes(elementCount, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := lookupHawkesLogLikelihoodKernel(benchmark, storageDType).Run(
					events, totalTime, baseline, alpha, beta, out,
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

func hawkesLogLikelihoodBenchmarkBytes(elementCount int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))

	return int64(elementCount)*elementBytes + elementBytes*4 + elementBytes
}
