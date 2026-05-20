package metal

import (
	"context"
	"testing"
)

func BenchmarkKernel_RunAdaptiveMaxPool2DDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	outputWidth := parityElementCounts[len(parityElementCounts)-1]

	for _, storageDType := range metalVisionDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			inputBytes, _ := adaptiveMaxPool2DDTypeBytes(outputWidth, storageDType)
			input, out := adaptiveMaxPool2DTensorsForTest(
				benchmark, backend, outputWidth, storageDType, inputBytes,
			)
			defer closeBenchmarkTensors(input, out)

			benchmark.SetBytes(adaptivePool2DBenchmarkBytes(outputWidth, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := runMetalAdaptiveMaxPool2D(input, out); err != nil {
					benchmark.Fatal(err)
				}

				if err := out.Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}
