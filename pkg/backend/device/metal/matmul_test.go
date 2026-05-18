package metal

import (
	"strconv"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func TestKernelRegistry_MetalMatMulDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalMatMulDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalMatMulDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalMatMulDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, inner := range parityElementCounts {
		inner := inner

		testingObject.Run(testNameForElementCount(inner), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for matmul", testingObject, func() {
				kernel := lookupMatMulKernel(testingObject, storageDType)
				runMatMulParityCase(testingObject, backend, kernel, storageDType, inner)
			})
		})
	}
}

func TestKernelRegistry_MetalMatMulAddDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalMatMulDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalMatMulAddDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalMatMulAddDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, inner := range parityElementCounts {
		inner := inner

		testingObject.Run(testNameForElementCount(inner), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for matmul_add", testingObject, func() {
				kernel := lookupMatMulAddKernel(testingObject, storageDType)
				runMatMulAddParityCase(testingObject, backend, kernel, storageDType, inner)
			})
		})
	}
}

func runMatMulParityCase(
	testingObject testing.TB,
	backend *Backend,
	kernel kernels.Kernel,
	storageDType dtype.DType,
	inner int,
) {
	rows, cols := 17, 19
	left, right, _, expected := matMulDTypeBytes(rows, inner, cols, storageDType, false)
	leftTensor, rightTensor, outTensor := matMulTensorsForTest(
		testingObject, backend, rows, inner, cols, storageDType, left, right,
	)
	defer closeBenchmarkTensors(leftTensor, rightTensor, outTensor)

	err := kernel.Run(leftTensor, rightTensor, outTensor)
	convey.So(err, convey.ShouldBeNil)
	assertMatMulBytesForTest(testingObject, backend, outTensor, storageDType, expected)
}

func runMatMulAddParityCase(
	testingObject testing.TB,
	backend *Backend,
	kernel kernels.Kernel,
	storageDType dtype.DType,
	inner int,
) {
	rows, cols := 17, 19
	left, right, bias, expected := matMulDTypeBytes(rows, inner, cols, storageDType, true)
	leftTensor, rightTensor, outTensor := matMulTensorsForTest(
		testingObject, backend, rows, inner, cols, storageDType, left, right,
	)
	biasTensor := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{cols}), storageDType, bias,
	)
	defer closeBenchmarkTensors(leftTensor, rightTensor, biasTensor, outTensor)

	err := kernel.Run(leftTensor, rightTensor, biasTensor, outTensor)
	convey.So(err, convey.ShouldBeNil)
	assertMatMulBytesForTest(testingObject, backend, outTensor, storageDType, expected)
}

func lookupMatMulKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("matmul", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s matmul kernel", storageDType.Name())
	}

	return kernel
}

func lookupMatMulAddKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("matmul_add", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s matmul_add kernel", storageDType.Name())
	}

	return kernel
}

func matMulTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	inner int,
	cols int,
	storageDType dtype.DType,
	leftBytes []byte,
	rightBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	leftShape := mustShapeForTest(testingObject, []int{rows, inner})
	rightShape := mustShapeForTest(testingObject, []int{inner, cols})
	outShape := mustShapeForTest(testingObject, []int{rows, cols})
	left := uploadDTypeTensorForTest(testingObject, backend, leftShape, storageDType, leftBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, rightShape, storageDType, rightBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	return left, right, out
}

func matMulDTypeBytes(
	rows int,
	inner int,
	cols int,
	storageDType dtype.DType,
	withBias bool,
) ([]byte, []byte, []byte, []byte) {
	leftValues, rightValues, biasValues := matMulInputValues(rows, inner, cols)
	leftBytes := encodeMatMulValuesAsDType(leftValues, storageDType)
	rightBytes := encodeMatMulValuesAsDType(rightValues, storageDType)
	biasBytes := encodeMatMulValuesAsDType(biasValues, storageDType)
	leftStored := decodeDTypeBytesToFloat32(leftBytes, storageDType)
	rightStored := decodeDTypeBytesToFloat32(rightBytes, storageDType)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, storageDType)
	expectedValues := matMulExpectedValues(
		leftStored, rightStored, biasStored, rows, inner, cols, withBias,
	)

	return leftBytes, rightBytes, biasBytes, encodeMatMulValuesAsDType(expectedValues, storageDType)
}

func matMulInputValues(rows int, inner int, cols int) ([]float32, []float32, []float32) {
	leftValues := make([]float32, rows*inner)
	rightValues := make([]float32, inner*cols)
	biasValues := make([]float32, cols)

	for index := range leftValues {
		leftValues[index] = centeredPowerOfTwoValue(index, 23, 32)
	}

	for index := range rightValues {
		rightValues[index] = centeredPowerOfTwoValue(index*3+1, 29, 64)
	}

	for index := range biasValues {
		biasValues[index] = centeredPowerOfTwoValue(index*5+2, 11, 16)
	}

	return leftValues, rightValues, biasValues
}

func centeredPowerOfTwoValue(index int, modulus int, divisor float32) float32 {
	centered := index%modulus - modulus/2
	return float32(centered) / divisor
}

func matMulExpectedValues(
	left []float32,
	right []float32,
	bias []float32,
	rows int,
	inner int,
	cols int,
	withBias bool,
) []float32 {
	out := make([]float32, rows*cols)

	for rowIndex := range rows {
		for colIndex := range cols {
			out[rowIndex*cols+colIndex] = matMulExpectedCell(
				left, right, bias, rowIndex, inner, colIndex, cols, withBias,
			)
		}
	}

	return out
}

func matMulExpectedCell(
	left []float32,
	right []float32,
	bias []float32,
	rowIndex int,
	inner int,
	colIndex int,
	cols int,
	withBias bool,
) float32 {
	accumulator := float32(0)
	if withBias {
		accumulator = bias[colIndex]
	}

	for innerIndex := range inner {
		accumulator += left[rowIndex*inner+innerIndex] * right[innerIndex*cols+colIndex]
	}

	return accumulator
}

func encodeMatMulValuesAsDType(values []float32, storageDType dtype.DType) []byte {
	if storageDType == dtype.Float32 {
		return dtypeconvert.Float32ToBytes(values)
	}

	return encodeFloat32ValuesAsDType(values, storageDType)
}

func assertMatMulBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
) {
	testingObject.Helper()

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
	expectedValues := decodeDTypeBytesToFloat32(expectedBytes, storageDType)
	assertFloat32WithinULP(testingObject, actualValues, expectedValues, 1)
}

func testNameForElementCount(elementCount int) string {
	return "N=" + strconv.Itoa(elementCount)
}
