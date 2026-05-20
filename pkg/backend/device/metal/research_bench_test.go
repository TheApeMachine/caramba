package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunResearchDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalResearchDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkVSAResearchDType(benchmark, backend, storageDType)
			benchmarkPredictiveCodingDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkVSAResearchDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, name := range []string{
		"vsa_bind",
		"vsa_bundle",
		"vsa_permute",
		"vsa_inverse_permute",
	} {
		name := name

		benchmark.Run(name, func(benchmark *testing.B) {
			tensors := benchmarkVSATensors(benchmark, backend, name, storageDType)
			defer closeBenchmarkTensors(tensors...)

			benchmark.SetBytes(int64(8192 * researchTensorCount(name) * dtypeBytesForBenchmark(storageDType)))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := benchmarkVSAKernel(benchmark, name, storageDType).Run(tensors...); err != nil {
					benchmark.Fatal(err)
				}

				if err := tensors[len(tensors)-1].Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func benchmarkPredictiveCodingDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, name := range []string{
		"pc_prediction",
		"pc_prediction_error",
		"pc_update_representation",
		"pc_update_weights",
	} {
		name := name

		benchmark.Run(name, func(benchmark *testing.B) {
			tensors := benchmarkPCTensors(benchmark, backend, name, storageDType)
			defer closeBenchmarkTensors(tensors...)

			benchmark.SetBytes(benchmarkPCBytes(name, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := benchmarkPCKernel(benchmark, name, storageDType).Run(tensors...); err != nil {
					benchmark.Fatal(err)
				}

				if err := tensors[len(tensors)-1].Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func benchmarkVSATensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) []tensor.Tensor {
	elementCount := 8192
	shape := mustShapeForTest(testingObject, []int{elementCount})

	if name == "vsa_permute" || name == "vsa_inverse_permute" {
		fixture := vsaUnaryFixtureForTest(name, storageDType, elementCount)
		input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.inputBytes)
		out := emptyTensorForTest(testingObject, backend, shape, storageDType)
		return []tensor.Tensor{input, out}
	}

	fixture := vsaBinaryFixtureForTest(name, storageDType, elementCount)
	left := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.leftBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.rightBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return []tensor.Tensor{left, right, out}
}

func benchmarkPCTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) []tensor.Tensor {
	outCount, inCount := 128, 1024

	if name == "pc_prediction_error" {
		fixture := pcPredictionErrorFixtureForTest(storageDType, inCount)
		shape := mustShapeForTest(testingObject, []int{inCount})
		observed := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.observedBytes)
		predicted := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.predictedBytes)
		out := emptyTensorForTest(testingObject, backend, shape, storageDType)
		return []tensor.Tensor{observed, predicted, out}
	}

	fixture := pcFixtureForTest(storageDType, outCount, inCount)
	if name == "pc_prediction" {
		weightShape := mustShapeForTest(testingObject, []int{outCount, inCount})
		stateShape := mustShapeForTest(testingObject, []int{inCount})
		outShape := mustShapeForTest(testingObject, []int{outCount})
		weights := uploadDTypeTensorForTest(testingObject, backend, weightShape, storageDType, fixture.weightBytes)
		state := uploadDTypeTensorForTest(testingObject, backend, stateShape, storageDType, fixture.stateBytes)
		out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
		return []tensor.Tensor{weights, state, out}
	}

	weightOutput := name == "pc_update_weights"
	weights, state, predictionError, out := pcUpdateTensorsForTest(
		testingObject, backend, storageDType, fixture, outCount, inCount, weightOutput,
	)
	return []tensor.Tensor{weights, state, predictionError, out}
}

func benchmarkVSAKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	if name == "vsa_permute" || name == "vsa_inverse_permute" {
		return lookupResearchUnaryKernel(testingObject, name, storageDType)
	}

	return lookupResearchBinaryKernel(testingObject, name, storageDType)
}

func benchmarkPCKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	if name == "pc_update_representation" || name == "pc_update_weights" {
		return lookupResearchTernaryKernel(testingObject, name, storageDType)
	}

	return lookupResearchBinaryKernel(testingObject, name, storageDType)
}

func researchTensorCount(name string) int {
	if name == "vsa_permute" || name == "vsa_inverse_permute" {
		return 2
	}

	return 3
}

func benchmarkPCBytes(name string, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))

	switch name {
	case "pc_prediction":
		return int64(128*1024+1024+128) * elementBytes
	case "pc_prediction_error":
		return int64(1024*3) * elementBytes
	case "pc_update_representation":
		return int64(128*1024+1024+128+1024) * elementBytes
	default:
		return int64(128*1024+1024+128+128*1024) * elementBytes
	}
}
