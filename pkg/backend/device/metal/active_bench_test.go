package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunActiveInferenceDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalActiveDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkActiveInferenceDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkActiveInferenceDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, name := range []string{
		"free_energy",
		"expected_free_energy",
		"belief_update",
		"precision_weight",
	} {
		name := name

		benchmark.Run(name, func(benchmark *testing.B) {
			tensors := benchmarkActiveTensors(benchmark, backend, name, storageDType)
			defer closeBenchmarkTensors(tensors...)

			benchmark.SetBytes(activeBenchmarkBytes(name, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := benchmarkActiveKernel(benchmark, name, storageDType).Run(tensors...); err != nil {
					benchmark.Fatal(err)
				}

				if err := tensors[len(tensors)-1].Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func benchmarkActiveTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) []tensor.Tensor {
	elementCount := 8192
	shape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := shape

	if name == "free_energy" {
		fixture := activeFreeEnergyFixtureForTest(storageDType, elementCount)
		outShape = mustShapeForTest(testingObject, []int{1})
		return benchmarkActiveFreeEnergyTensors(testingObject, backend, storageDType, shape, outShape, fixture)
	}

	if name == "expected_free_energy" {
		fixture := activeExpectedFreeEnergyFixtureForTest(storageDType, elementCount, elementCount)
		outShape = mustShapeForTest(testingObject, []int{1})
		return benchmarkActiveExpectedFreeEnergyTensors(testingObject, backend, storageDType, shape, outShape, fixture)
	}

	fixture := activeBinaryFixtureForTest(name, storageDType, elementCount)
	left := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	return []tensor.Tensor{left, right, out}
}

func benchmarkActiveFreeEnergyTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	shape tensor.Shape,
	outShape tensor.Shape,
	fixture activeFixture,
) []tensor.Tensor {
	likelihood := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	posterior := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	prior := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.thirdBytes)
	auxiliary := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.auxiliaryBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	return []tensor.Tensor{likelihood, posterior, prior, auxiliary, out}
}

func benchmarkActiveExpectedFreeEnergyTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	shape tensor.Shape,
	outShape tensor.Shape,
	fixture activeFixture,
) []tensor.Tensor {
	predictedObs := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	preferredObs := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	predictedState := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.thirdBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	return []tensor.Tensor{predictedObs, preferredObs, predictedState, out}
}

func benchmarkActiveKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	switch name {
	case "free_energy":
		return lookupActiveFreeEnergyKernel(testingObject, storageDType)
	case "expected_free_energy":
		return lookupActiveExpectedFreeEnergyKernel(testingObject, storageDType)
	default:
		return lookupActiveBinaryKernel(testingObject, name, storageDType)
	}
}

func activeBenchmarkBytes(name string, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))

	switch name {
	case "free_energy":
		return int64(8192*4+1) * elementBytes
	case "expected_free_energy":
		return int64(8192*3+1) * elementBytes
	default:
		return int64(8192*3) * elementBytes
	}
}
