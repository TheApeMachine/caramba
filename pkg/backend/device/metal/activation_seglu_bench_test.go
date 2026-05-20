package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func BenchmarkKernel_RunSeGLUDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalSeGLUDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkSeGLUDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkSeGLUDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	elementCount := parityElementCounts[len(parityElementCounts)-1]
	shape := mustShapeForTest(benchmark, []int{elementCount})
	fixture := segluFixtureForTest(elementCount, storageDType)
	gate := uploadDTypeTensorForTest(benchmark, backend, shape, storageDType, fixture.gateBytes)
	up := uploadDTypeTensorForTest(benchmark, backend, shape, storageDType, fixture.upBytes)
	out := emptyTensorForTest(benchmark, backend, shape, storageDType)
	defer closeBenchmarkTensors(gate, up, out)

	elementBytes := dtypeBytesForBenchmark(storageDType)
	benchmark.SetBytes(int64(shape.Len() * 3 * elementBytes))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		err := runMetalSeGLU(gate, up, out)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}
