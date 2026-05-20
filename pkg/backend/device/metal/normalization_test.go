package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

const normalizationFloat32MaxULP = 32

func TestKernelRegistry_MetalLayerNormDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalNormalizationDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalLayerNormDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalLayerNormDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, cols := range parityElementCounts {
		cols := cols

		testingObject.Run(testNameForElementCount(cols), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for layernorm", testingObject, func() {
				kernel := lookupLayerNormKernel(testingObject, storageDType)
				runLayerNormParityCase(testingObject, backend, kernel, storageDType, cols)
			})
		})
	}
}

func TestKernelRegistry_MetalRMSNormDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalNormalizationDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalRMSNormDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalRMSNormDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, cols := range parityElementCounts {
		cols := cols

		testingObject.Run(testNameForElementCount(cols), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for rmsnorm", testingObject, func() {
				kernel := lookupRMSNormKernel(testingObject, storageDType)
				runRMSNormParityCase(testingObject, backend, kernel, storageDType, cols)
			})
		})
	}
}

func runLayerNormParityCase(
	testingObject testing.TB,
	backend *Backend,
	kernel kernels.Kernel,
	storageDType dtype.DType,
	cols int,
) {
	rows := 13
	inputBytes, scaleBytes, biasBytes := normDTypeBytes(rows, cols, storageDType)
	expectedBytes := expectedLayerNormBytesForTest(rows, cols, storageDType)
	input, scale, bias, out := layerNormTensorsForTest(
		testingObject, backend, rows, cols, storageDType, inputBytes, scaleBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, scale, bias, out)

	err := kernel.Run(input, scale, bias, out)
	convey.So(err, convey.ShouldBeNil)
	assertNormalizationBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func runRMSNormParityCase(
	testingObject testing.TB,
	backend *Backend,
	kernel kernels.Kernel,
	storageDType dtype.DType,
	cols int,
) {
	rows := 13
	inputBytes, scaleBytes, _ := normDTypeBytes(rows, cols, storageDType)
	expectedBytes := expectedRMSNormBytesForTest(rows, cols, storageDType)
	input, scale, out := rmsNormTensorsForTest(
		testingObject, backend, rows, cols, storageDType, inputBytes, scaleBytes,
	)
	defer closeBenchmarkTensors(input, scale, out)

	err := kernel.Run(input, scale, out)
	convey.So(err, convey.ShouldBeNil)
	assertNormalizationBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func lookupLayerNormKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("layernorm", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s layernorm kernel", storageDType.Name())
	}

	return kernel
}

func lookupRMSNormKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("rmsnorm", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s rmsnorm kernel", storageDType.Name())
	}

	return kernel
}

func layerNormTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	cols int,
	storageDType dtype.DType,
	inputBytes []byte,
	scaleBytes []byte,
	biasBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{rows, cols})
	paramShape := mustShapeForTest(testingObject, []int{cols})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, inputBytes)
	scale := uploadDTypeTensorForTest(testingObject, backend, paramShape, storageDType, scaleBytes)
	bias := uploadDTypeTensorForTest(testingObject, backend, paramShape, storageDType, biasBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return input, scale, bias, out
}

func rmsNormTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	cols int,
	storageDType dtype.DType,
	inputBytes []byte,
	scaleBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{rows, cols})
	paramShape := mustShapeForTest(testingObject, []int{cols})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, inputBytes)
	scale := uploadDTypeTensorForTest(testingObject, backend, paramShape, storageDType, scaleBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return input, scale, out
}

func normDTypeBytes(
	rows int,
	cols int,
	storageDType dtype.DType,
) ([]byte, []byte, []byte) {
	inputValues, scaleValues, biasValues := normInputValues(rows, cols)
	inputBytes := encodeNormValuesAsDType(inputValues, storageDType)
	scaleBytes := encodeNormValuesAsDType(scaleValues, storageDType)
	biasBytes := encodeNormValuesAsDType(biasValues, storageDType)
	return inputBytes, scaleBytes, biasBytes
}

func normInputValues(rows int, cols int) ([]float32, []float32, []float32) {
	inputValues := make([]float32, rows*cols)
	scaleValues := make([]float32, cols)
	biasValues := make([]float32, cols)

	for index := range inputValues {
		inputValues[index] = centeredPowerOfTwoValue(index*7+3, 41, 16)
	}

	for index := range scaleValues {
		scaleValues[index] = 1 + centeredPowerOfTwoValue(index*5+1, 17, 64)
		biasValues[index] = centeredPowerOfTwoValue(index*11+2, 19, 128)
	}

	return inputValues, scaleValues, biasValues
}
