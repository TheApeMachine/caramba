package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkKernel_RunLinGLUDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalLinGLUDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkLinGLUDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkLinGLUDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	elementCount := parityElementCounts[len(parityElementCounts)-1]
	shape := mustShapeForTest(benchmark, []int{elementCount})
	fixture := lingluFixtureForTest(elementCount, storageDType)
	gate := uploadDTypeTensorForTest(benchmark, backend, shape, storageDType, fixture.gateBytes)
	up := uploadDTypeTensorForTest(benchmark, backend, shape, storageDType, fixture.upBytes)
	out := emptyTensorForTest(benchmark, backend, shape, storageDType)
	defer closeBenchmarkTensors(gate, up, out)

	elementBytes := dtypeBytesForBenchmark(storageDType)
	benchmark.SetBytes(int64(shape.Len() * 3 * elementBytes))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		err := runMetalLinGLU(gate, up, out)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}
