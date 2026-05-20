package metal

import (
	"context"
	"testing"
)

func BenchmarkKernel_RunConvTranspose2DDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	width := parityElementCounts[len(parityElementCounts)-1]

	for _, storageDType := range metalVisionDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			inputWidth := convTransposeInputWidthForTest(width)
			inputBytes, weightBytes, biasBytes, _ :=
				convTranspose2DDTypeBytes(width, storageDType)
			input, weight, bias, out := convTranspose2DTensorsForTest(
				benchmark, backend, inputWidth, storageDType,
				inputBytes, weightBytes, biasBytes,
			)
			defer closeBenchmarkTensors(input, weight, bias, out)

			benchmark.SetBytes(convTranspose2DBenchmarkBytes(inputWidth, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := runMetalConvTranspose2D(input, weight, bias, out); err != nil {
					benchmark.Fatal(err)
				}

				if err := out.Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}
