package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunPhysicsDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalPhysicsDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkPhysicsDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkPhysicsDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, name := range physicsBenchmarkNames() {
		name := name

		benchmark.Run(name, func(benchmark *testing.B) {
			tensors := benchmarkPhysicsTensors(benchmark, backend, name, storageDType)
			defer closeBenchmarkTensors(tensors...)

			benchmark.SetBytes(physicsBenchmarkBytes(name, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := benchmarkPhysicsKernel(benchmark, name, storageDType).Run(tensors...); err != nil {
					benchmark.Fatal(err)
				}

				if err := tensors[len(tensors)-1].Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func physicsBenchmarkNames() []string {
	return []string{
		"laplacian",
		"laplacian4",
		"grad1d",
		"divergence1d",
		"fft1d",
		"ifft1d",
		"quantum_potential",
		"bohmian_velocity",
		"madelung_continuity",
	}
}

func benchmarkPhysicsTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) []tensor.Tensor {
	if name == "fft1d" || name == "ifft1d" {
		return benchmarkPhysicsFFTTensors(testingObject, backend, storageDType, name == "ifft1d")
	}

	if name == "madelung_continuity" {
		return benchmarkMadelungTensors(testingObject, backend, storageDType)
	}

	return benchmarkPhysicsUnaryTensors(testingObject, backend, name, storageDType)
}

func benchmarkPhysicsUnaryTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) []tensor.Tensor {
	dims := []int{8192}
	if name == "laplacian" {
		dims = []int{8, 32, 32}
	}

	fixture := physicsUnaryFixtureForTest(name, dims, storageDType)
	shape := mustShapeForTest(testingObject, dims)
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.inputBytes)
	spacing := uploadDTypeTensorForTest(
		testingObject, backend, scalarShapeForTest(testingObject), storageDType, fixture.spacingBytes,
	)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return []tensor.Tensor{input, spacing, out}
}

func benchmarkMadelungTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	elementCount := 8192
	fixture := madelungFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	density := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	velocity := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	spacing := uploadDTypeTensorForTest(
		testingObject, backend, scalarShapeForTest(testingObject), storageDType, fixture.spacingBytes,
	)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return []tensor.Tensor{density, velocity, spacing, out}
}

func benchmarkPhysicsFFTTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	inverse bool,
) []tensor.Tensor {
	elementCount := 1024
	fixture := physicsFFTFixtureForTest(elementCount, storageDType, inverse)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	realIn := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.realInBytes)
	imagIn := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.imagInBytes)
	realOut := emptyTensorForTest(testingObject, backend, shape, storageDType)
	imagOut := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return []tensor.Tensor{realIn, imagIn, realOut, imagOut}
}

func benchmarkPhysicsKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	if name == "fft1d" || name == "ifft1d" {
		return lookupPhysicsFFTKernel(testingObject, name, storageDType)
	}

	if name == "madelung_continuity" {
		return lookupMadelungKernel(testingObject, storageDType)
	}

	return lookupPhysicsBinaryKernel(testingObject, name, storageDType)
}

func physicsBenchmarkBytes(name string, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))

	switch name {
	case "fft1d", "ifft1d":
		return int64(1024*4) * elementBytes
	case "madelung_continuity":
		return int64(8192*3+1) * elementBytes
	default:
		return int64(8192*2+1) * elementBytes
	}
}
