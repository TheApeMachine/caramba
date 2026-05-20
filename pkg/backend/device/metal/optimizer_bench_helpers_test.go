package metal

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func optimizer3BenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) []tensor.Tensor {
	paramBytes, gradientBytes, _, _ := optimizerStorageInputs(elementCount, storageDType)
	stateValues := optimizerStateValues(elementCount, 7)

	return optimizer3TensorsForTest(
		testingObject, backend, storageDType, elementCount,
		paramBytes, gradientBytes, stateValues,
	)
}

func optimizer2BenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) []tensor.Tensor {
	paramBytes, gradientBytes, _, _ := optimizerStorageInputs(elementCount, storageDType)

	return optimizer2TensorsForTest(
		testingObject, backend, storageDType, elementCount, paramBytes, gradientBytes,
	)
}

func optimizer4BenchmarkBytes(
	testingObject testing.TB,
	elementCount int,
	storageDType dtype.DType,
) int64 {
	elementBytes := dtypeSizeForTest(testingObject, storageDType)
	return int64(elementCount * (3*elementBytes + 8))
}

func optimizer3BenchmarkBytes(
	testingObject testing.TB,
	elementCount int,
	storageDType dtype.DType,
) int64 {
	elementBytes := dtypeSizeForTest(testingObject, storageDType)
	return int64(elementCount * (3*elementBytes + 4))
}

func optimizer2BenchmarkBytes(
	testingObject testing.TB,
	elementCount int,
	storageDType dtype.DType,
) int64 {
	elementBytes := dtypeSizeForTest(testingObject, storageDType)
	return int64(elementCount * 3 * elementBytes)
}

func optimizerHebbianBenchmarkBytes(
	testingObject testing.TB,
	elementCount int,
	storageDType dtype.DType,
) int64 {
	elementBytes := dtypeSizeForTest(testingObject, storageDType)
	return int64((3*elementCount + 1) * elementBytes)
}
