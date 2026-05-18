package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

const softmaxFloat32MaxULP = 64

func TestKernelRegistry_MetalSoftmaxDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalSoftmaxDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalSoftmaxDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalSoftmaxDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, cols := range parityElementCounts {
		cols := cols

		testingObject.Run(testNameForElementCount(cols), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for softmax", testingObject, func() {
				kernel := lookupSoftmaxKernel(testingObject, storageDType)
				runSoftmaxParityCase(testingObject, backend, kernel, storageDType, cols)
			})
		})
	}
}

func runSoftmaxParityCase(
	testingObject testing.TB,
	backend *Backend,
	kernel kernels.Kernel,
	storageDType dtype.DType,
	cols int,
) {
	rows := 13
	inputBytes, expectedBytes := softmaxDTypeBytes(rows, cols, storageDType)
	inputTensor, outTensor := softmaxTensorsForTest(
		testingObject, backend, rows, cols, storageDType, inputBytes,
	)
	defer closeBenchmarkTensors(inputTensor, outTensor)

	err := kernel.Run(inputTensor, outTensor)
	convey.So(err, convey.ShouldBeNil)
	assertSoftmaxBytesForTest(testingObject, backend, outTensor, storageDType, expectedBytes)
}

func lookupSoftmaxKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("softmax", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s softmax kernel", storageDType.Name())
	}

	return kernel
}

func softmaxTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	rows int,
	cols int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{rows, cols})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return input, out
}

func softmaxDTypeBytes(
	rows int,
	cols int,
	storageDType dtype.DType,
) ([]byte, []byte) {
	inputValues := softmaxInputValues(rows, cols)
	inputBytes := encodeSoftmaxValuesAsDType(inputValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expectedValues := softmaxExpectedValues(inputStored, rows, cols)

	return inputBytes, encodeSoftmaxValuesAsDType(expectedValues, storageDType)
}

func softmaxInputValues(rows int, cols int) []float32 {
	values := make([]float32, rows*cols)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*7+3, 37, 8)
	}

	return values
}

func softmaxExpectedValues(input []float32, rows int, cols int) []float32 {
	out := make([]float32, len(input))

	for rowIndex := range rows {
		rowOffset := rowIndex * cols
		row := input[rowOffset : rowOffset+cols]
		outRow := out[rowOffset : rowOffset+cols]

		maximum := findSoftmaxRowMax(row)
		sum := fillSoftmaxExpectedRow(row, outRow, maximum)
		normalizeSoftmaxExpectedRow(outRow, sum)
	}

	return out
}

func findSoftmaxRowMax(row []float32) float32 {
	maximum := row[0]

	for _, candidate := range row[1:] {
		if candidate > maximum {
			maximum = candidate
		}
	}

	return maximum
}

func fillSoftmaxExpectedRow(row []float32, outRow []float32, maximum float32) float32 {
	var sum float32

	for index, candidate := range row {
		shifted := float32(math.Exp(float64(candidate - maximum)))
		outRow[index] = shifted
		sum += shifted
	}

	return sum
}

func normalizeSoftmaxExpectedRow(outRow []float32, sum float32) {
	if sum == 0 {
		return
	}

	for index := range outRow {
		outRow[index] /= sum
	}
}

func encodeSoftmaxValuesAsDType(values []float32, storageDType dtype.DType) []byte {
	if storageDType == dtype.Float32 {
		return dtypeconvert.Float32ToBytes(values)
	}

	return encodeFloat32ValuesAsDType(values, storageDType)
}

func assertSoftmaxBytesForTest(
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
	assertSoftmaxFloat32WithinULP(
		testingObject,
		actualValues,
		expectedValues,
		softmaxFloat32MaxULP,
	)
}

func assertSoftmaxFloat32WithinULP(
	testingObject testing.TB,
	actualValues []float32,
	expectedValues []float32,
	maxULP uint32,
) {
	testingObject.Helper()

	if len(actualValues) != len(expectedValues) {
		testingObject.Fatalf("length mismatch: got %d want %d", len(actualValues), len(expectedValues))
	}

	maxDistance, maxIndex := maxSoftmaxFloat32ULPDistance(actualValues, expectedValues)
	if maxDistance <= maxULP {
		return
	}

	testingObject.Fatalf(
		"softmax float32 max ULP mismatch at %d: got %08x (%g), want %08x (%g), distance %d > %d",
		maxIndex,
		math.Float32bits(actualValues[maxIndex]),
		actualValues[maxIndex],
		math.Float32bits(expectedValues[maxIndex]),
		expectedValues[maxIndex],
		maxDistance,
		maxULP,
	)
}

func maxSoftmaxFloat32ULPDistance(
	actualValues []float32,
	expectedValues []float32,
) (uint32, int) {
	var maxDistance uint32
	var maxIndex int

	for index := range actualValues {
		distance := float32ULPDistance(actualValues[index], expectedValues[index])
		if distance <= maxDistance {
			continue
		}

		maxDistance = distance
		maxIndex = index
	}

	return maxDistance, maxIndex
}
