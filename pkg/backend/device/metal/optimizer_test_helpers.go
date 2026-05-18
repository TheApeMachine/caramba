package metal

import (
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func optimizer2TensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	paramBytes []byte,
	gradientBytes []byte,
) []tensor.Tensor {
	shape := mustShapeForTest(testingObject, []int{elementCount})
	params := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, paramBytes)
	gradients := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, gradientBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)

	return []tensor.Tensor{params, gradients, out}
}

func hebbianTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	weightBytes []byte,
	postBytes []byte,
	preBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	weightShape := mustShapeForTest(testingObject, []int{1, elementCount})
	vectorShape := mustShapeForTest(testingObject, []int{elementCount})
	postShape := mustShapeForTest(testingObject, []int{1})
	weights := uploadDTypeTensorForTest(
		testingObject, backend, weightShape, storageDType, weightBytes,
	)
	post := uploadDTypeTensorForTest(testingObject, backend, postShape, storageDType, postBytes)
	pre := uploadDTypeTensorForTest(testingObject, backend, vectorShape, storageDType, preBytes)
	out := emptyTensorForTest(testingObject, backend, weightShape, storageDType)

	return weights, post, pre, out
}

func hebbianDTypeBytes(
	elementCount int,
	storageDType dtype.DType,
) ([]byte, []byte, []byte, []float32) {
	weightValues := projectionValues(elementCount, 73, 64)
	postValues := []float32{0.25}
	preValues := projectionValues(elementCount, 79, 128)
	weightBytes := encodeProjectionValuesAsDType(weightValues, storageDType)
	postBytes := encodeProjectionValuesAsDType(postValues, storageDType)
	preBytes := encodeProjectionValuesAsDType(preValues, storageDType)
	weights := decodeDTypeBytesToFloat32(weightBytes, storageDType)
	post := decodeDTypeBytesToFloat32(postBytes, storageDType)
	pre := decodeDTypeBytesToFloat32(preBytes, storageDType)

	return weightBytes, postBytes, preBytes, optimizerHebbianExpected(weights, post, pre)
}

func lookupOptimizer4Kernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, dtype.Float32, dtype.Float32,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s optimizer kernel for %s", storageDType.Name(), name)
	}

	return kernel
}

func lookupOptimizer3Kernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, dtype.Float32,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s optimizer kernel for %s", storageDType.Name(), name)
	}

	return kernel
}

func lookupOptimizer2Kernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s optimizer kernel for %s", storageDType.Name(), name)
	}

	return kernel
}

func lookupHebbianKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("hebbian_step", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s hebbian_step kernel", storageDType.Name())
	}

	return kernel
}

func assertOptimizerStorageForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedValues []float32,
	maxULP uint32,
) {
	testingObject.Helper()

	expectedBytes := encodeProjectionValuesAsDType(expectedValues, storageDType)
	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, expectedBytes, 1)
		return
	}

	actualDType, actualBytes, err := backend.Download(input)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	if actualDType != storageDType {
		testingObject.Fatalf("download dtype mismatch: got %s want %s", actualDType, storageDType)
	}

	actualValues := decodeDTypeBytesToFloat32(actualBytes, storageDType)
	expectedStored := decodeDTypeBytesToFloat32(expectedBytes, storageDType)
	assertFloat32WithinULP(testingObject, actualValues, expectedStored, maxULP)
}

func configureOptimizerExpectedArithmetic(storageDType dtype.DType) func() {
	previous := optimizerExpectedUsesFMA
	optimizerExpectedUsesFMA = storageDType != dtype.BFloat16

	return func() {
		optimizerExpectedUsesFMA = previous
	}
}

func assertOptimizerStateForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	expectedValues []float32,
) {
	testingObject.Helper()

	actualDType, actualBytes, err := backend.Download(input)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	if actualDType != dtype.Float32 {
		testingObject.Fatalf("download dtype mismatch: got %s want %s", actualDType, dtype.Float32)
	}

	actualValues := decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
	assertFloat32WithinULP(testingObject, actualValues, expectedValues, optimizerMaxULP)
}
