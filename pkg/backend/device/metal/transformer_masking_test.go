package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalMaskingAndPositionalDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalTransformerDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalMaskingAndPositionalDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalMaskingAndPositionalDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" masking tensors", testingObject, func() {
				runApplyMaskParityCase(testingObject, backend, storageDType, elementCount)
				runCausalMaskParityCase(testingObject, backend, storageDType, elementCount)
				runALiBiBiasParityCase(testingObject, backend, storageDType, elementCount)
			})
		})
	}
}

func runApplyMaskParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inputBytes, maskBytes, expectedBytes := applyMaskDTypeBytes(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, inputBytes)
	mask := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, maskBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(input, mask, out)

	err := lookupApplyMaskKernel(testingObject, storageDType).Run(input, mask, out)
	convey.So(err, convey.ShouldBeNil)
	assertRawOrDTypeBytesForTest(testingObject, backend, out, storageDType, expectedBytes, 0)
}

func runCausalMaskParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	cols int,
) {
	rows := 7
	inputBytes := encodeProjectionValuesAsDType([]float32{0}, storageDType)
	expectedBytes := causalMaskDTypeBytes(rows, cols, storageDType)
	input := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{1}),
		storageDType, inputBytes,
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{rows, cols}),
		storageDType,
	)
	defer closeBenchmarkTensors(input, out)

	err := lookupCausalMaskKernel(testingObject, storageDType).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertRawOrDTypeBytesForTest(testingObject, backend, out, storageDType, expectedBytes, 0)
}

func runALiBiBiasParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	cols int,
) {
	rows := 7
	scoresBytes, slopeBytes, expectedBytes := alibiBiasDTypeBytes(rows, cols, storageDType)
	shape := mustShapeForTest(testingObject, []int{rows, cols})
	scores := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, scoresBytes)
	slope := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{1}),
		storageDType, slopeBytes,
	)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(scores, slope, out)

	err := lookupALiBiBiasKernel(testingObject, storageDType).Run(scores, slope, out)
	convey.So(err, convey.ShouldBeNil)
	assertRawOrDTypeBytesForTest(testingObject, backend, out, storageDType, expectedBytes, 0)
}

func lookupApplyMaskKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("apply_mask", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s apply_mask kernel", storageDType.Name())
	}

	return kernel
}

func lookupCausalMaskKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("causal_mask", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s causal_mask kernel", storageDType.Name())
	}

	return kernel
}

func lookupALiBiBiasKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("alibi_bias", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s alibi_bias kernel", storageDType.Name())
	}

	return kernel
}

func applyMaskDTypeBytes(elementCount int, storageDType dtype.DType) ([]byte, []byte, []byte) {
	inputValues := projectionValues(elementCount, 37, 64)
	maskValues := projectionValues(elementCount, 23, 32)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	maskBytes := encodeProjectionValuesAsDType(maskValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	maskStored := decodeDTypeBytesToFloat32(maskBytes, storageDType)
	expectedValues := make([]float32, elementCount)

	for index := range expectedValues {
		expectedValues[index] = inputStored[index] + maskStored[index]
	}

	return inputBytes, maskBytes, encodeProjectionValuesAsDType(expectedValues, storageDType)
}

func causalMaskDTypeBytes(rows int, cols int, storageDType dtype.DType) []byte {
	values := make([]float32, rows*cols)

	for rowIndex := range rows {
		for colIndex := range cols {
			value := float32(0)
			if colIndex > rowIndex {
				value = float32(math.Inf(-1))
			}

			values[rowIndex*cols+colIndex] = value
		}
	}

	return encodeProjectionValuesAsDType(values, storageDType)
}

func alibiBiasDTypeBytes(rows int, cols int, storageDType dtype.DType) ([]byte, []byte, []byte) {
	scoreValues := projectionValues(rows*cols, 41, 64)
	slopeValues := []float32{0.125}
	scoresBytes := encodeProjectionValuesAsDType(scoreValues, storageDType)
	slopeBytes := encodeProjectionValuesAsDType(slopeValues, storageDType)
	scoresStored := decodeDTypeBytesToFloat32(scoresBytes, storageDType)
	slopeStored := decodeDTypeBytesToFloat32(slopeBytes, storageDType)
	expectedValues := alibiBiasExpectedValues(scoresStored, slopeStored[0], rows, cols)

	return scoresBytes, slopeBytes, encodeProjectionValuesAsDType(expectedValues, storageDType)
}

func alibiBiasExpectedValues(
	scores []float32,
	slope float32,
	rows int,
	cols int,
) []float32 {
	out := make([]float32, len(scores))

	for rowIndex := range rows {
		for colIndex := range cols {
			index := rowIndex*cols + colIndex
			out[index] = scores[index]

			if rowIndex >= colIndex {
				out[index] -= slope * float32(rowIndex-colIndex)
			}
		}
	}

	return out
}
