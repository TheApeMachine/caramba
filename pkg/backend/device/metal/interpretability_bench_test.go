package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func BenchmarkKernel_RunActivationSteerFloat32(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	elementCount := 8192
	fixture := activationSteerFixtureForTest(elementCount)
	vectorShape := mustShapeForTest(benchmark, []int{elementCount})
	coefficientShape := mustShapeForTest(benchmark, []int{1})
	base := uploadDTypeTensorForTest(benchmark, backend, vectorShape, dtype.Float32, fixture.baseBytes)
	direction := uploadDTypeTensorForTest(benchmark, backend, vectorShape, dtype.Float32, fixture.directionBytes)
	coefficient := uploadDTypeTensorForTest(
		benchmark, backend, coefficientShape, dtype.Float32, fixture.coefficientBytes,
	)
	out := emptyTensorForTest(benchmark, backend, vectorShape, dtype.Float32)
	defer closeBenchmarkTensors(base, direction, coefficient, out)

	kernel := lookupActivationSteerFloat32Kernel(benchmark)
	benchmark.SetBytes(int64(elementCount * 12))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := kernel.Run(base, direction, coefficient, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}
