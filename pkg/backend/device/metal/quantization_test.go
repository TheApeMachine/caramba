package metal

import (
	"bytes"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalQuantization(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal quantization tensors", testingObject, func() {
				runInt8DequantParityCase(testingObject, backend, elementCount)
				runInt4DequantParityCase(testingObject, backend, elementCount)
				runInt8QuantParityCase(testingObject, backend, elementCount)
			})
		})
	}
}

func runInt8DequantParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) {
	inputBytes, expectedValues := int8DequantBytes(elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Int8, inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, dtype.Float32)
	defer closeBenchmarkTensors(input, out)

	err := lookupQuantizationKernel(testingObject, "int8_dequant", dtype.Int8, dtype.Float32).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertFloat32TensorForTest(testingObject, backend, out, expectedValues, 0)
}

func runInt4DequantParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) {
	inputBytes, expectedValues := int4DequantBytes(elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Int4, inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, dtype.Float32)
	defer closeBenchmarkTensors(input, out)

	err := lookupQuantizationKernel(testingObject, "int4_dequant", dtype.Int4, dtype.Float32).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertFloat32TensorForTest(testingObject, backend, out, expectedValues, 0)
}

func runInt8QuantParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) {
	inputValues, expectedBytes := int8QuantValues(elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(
		testingObject, backend, shape, dtype.Float32, dtypeconvert.Float32ToBytes(inputValues),
	)
	out := emptyTensorForTest(testingObject, backend, shape, dtype.Int8)
	defer closeBenchmarkTensors(input, out)

	err := lookupQuantizationKernel(testingObject, "int8_quant", dtype.Float32, dtype.Int8).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertRawDTypeBytesForTest(testingObject, backend, out, dtype.Int8, expectedBytes)
}

func lookupQuantizationKernel(
	testingObject testing.TB,
	name string,
	inputDType dtype.DType,
	outDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{inputDType},
		Outputs: []dtype.DType{outDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal quantization kernel for %s", name)
	}

	return kernel
}

func int8DequantBytes(elementCount int) ([]byte, []float32) {
	inputBytes := make([]byte, elementCount)
	expected := make([]float32, elementCount)

	for index := range inputBytes {
		value := int8((index*37)%256 - 128)
		inputBytes[index] = byte(value)
		expected[index] = float32(value)
	}

	return inputBytes, expected
}

func int4DequantBytes(elementCount int) ([]byte, []float32) {
	inputBytes := make([]byte, (elementCount+1)/2)
	expected := make([]float32, elementCount)

	for pairIndex := range inputBytes {
		lowIndex := pairIndex * 2
		highIndex := lowIndex + 1
		low := int8((lowIndex*5)%16 - 8)
		high := int8((highIndex*5)%16 - 8)
		inputBytes[pairIndex] = dtype.NewInt4Pair(low, high).Bits()
		expected[lowIndex] = float32(low)

		if highIndex < elementCount {
			expected[highIndex] = float32(high)
		}
	}

	return inputBytes, expected
}

func int8QuantValues(elementCount int) ([]float32, []byte) {
	inputValues := make([]float32, elementCount)
	expected := make([]byte, elementCount)

	for index := range inputValues {
		value := float32((index*41)%301-150) + 0.25
		inputValues[index] = value
		rounded := math.Round(float64(value))
		rounded = min(127, max(-128, rounded))
		expected[index] = byte(int8(rounded))
	}

	return inputValues, expected
}

func assertFloat32TensorForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	expectedValues []float32,
	maxULP uint32,
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
	assertFloat32WithinULP(testingObject, actualValues, expectedValues, maxULP)
}

func assertRawDTypeBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	expectedDType dtype.DType,
	expectedBytes []byte,
) {
	testingObject.Helper()

	actualDType, actualBytes, err := backend.Download(input)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	if actualDType != expectedDType {
		testingObject.Fatalf("download dtype mismatch: got %s want %s", actualDType, expectedDType)
	}

	if !bytes.Equal(actualBytes, expectedBytes) {
		testingObject.Fatalf("raw bytes mismatch: got %x want %x", actualBytes, expectedBytes)
	}
}
