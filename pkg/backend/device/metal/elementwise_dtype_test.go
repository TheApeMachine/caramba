package metal

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

var elementwiseStorageDTypes = []dtype.DType{dtype.Float16, dtype.BFloat16}

func TestKernelRegistry_MetalBinaryElementwiseDTypes(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range elementwiseStorageDTypes {
		storageDType := storageDType

		t.Run(storageDType.Name(), func(t *testing.T) {
			testKernelRegistryMetalBinaryElementwiseDType(t, backend, storageDType)
		})
	}
}

func testKernelRegistryMetalBinaryElementwiseDType(
	t *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, testCase := range binaryFloat32Cases {
		testCase := testCase

		t.Run(testCase.name, func(t *testing.T) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				t.Run(fmt.Sprintf("N=%d", elementCount), func(t *testing.T) {
					convey.Convey("Given Metal "+storageDType.Name()+" tensors for "+testCase.name, t, func() {
						kernel := lookupBinaryElementwiseKernel(t, testCase.name, storageDType)
						shape, err := tensor.NewShape([]int{elementCount})
						convey.So(err, convey.ShouldBeNil)

						leftBytes, rightBytes, expectedBytes := binaryElementwiseDTypeBytes(
							elementCount,
							testCase.name,
							storageDType,
						)

						left := uploadDTypeTensorForTest(t, backend, shape, storageDType, leftBytes)
						defer func() {
							convey.So(left.Close(), convey.ShouldBeNil)
						}()

						right := uploadDTypeTensorForTest(t, backend, shape, storageDType, rightBytes)
						defer func() {
							convey.So(right.Close(), convey.ShouldBeNil)
						}()

						out, err := backend.bridge.empty(shape, storageDType)
						convey.So(err, convey.ShouldBeNil)
						defer func() {
							convey.So(out.Close(), convey.ShouldBeNil)
						}()

						err = kernel.Run(left, right, out)
						convey.So(err, convey.ShouldBeNil)
						assertDTypeBytesForTest(t, backend, out, storageDType, expectedBytes, testCase.dtypeULP)
					})
				})
			}
		})
	}
}

func TestKernelRegistry_MetalUnaryElementwiseDTypes(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range elementwiseStorageDTypes {
		storageDType := storageDType

		t.Run(storageDType.Name(), func(t *testing.T) {
			testKernelRegistryMetalUnaryElementwiseDType(t, backend, storageDType)
		})
	}
}

func testKernelRegistryMetalUnaryElementwiseDType(
	t *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, testCase := range unaryFloat32Cases {
		testCase := testCase

		t.Run(testCase.name, func(t *testing.T) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				t.Run(fmt.Sprintf("N=%d", elementCount), func(t *testing.T) {
					convey.Convey("Given one Metal "+storageDType.Name()+" tensor for "+testCase.name, t, func() {
						kernel := lookupUnaryElementwiseKernel(t, testCase.name, storageDType)
						shape, err := tensor.NewShape([]int{elementCount})
						convey.So(err, convey.ShouldBeNil)

						inputBytes, expectedBytes := unaryElementwiseDTypeBytes(
							elementCount,
							testCase.name,
							storageDType,
						)

						input := uploadDTypeTensorForTest(t, backend, shape, storageDType, inputBytes)
						defer func() {
							convey.So(input.Close(), convey.ShouldBeNil)
						}()

						out, err := backend.bridge.empty(shape, storageDType)
						convey.So(err, convey.ShouldBeNil)
						defer func() {
							convey.So(out.Close(), convey.ShouldBeNil)
						}()

						err = kernel.Run(input, out)
						convey.So(err, convey.ShouldBeNil)
						assertDTypeBytesForTest(t, backend, out, storageDType, expectedBytes, testCase.maxULP)
					})
				})
			}
		})
	}
}

func lookupBinaryElementwiseKernel(
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
		testingObject.Fatalf("missing Metal %s binary kernel for %s", storageDType.Name(), name)
	}

	return kernel
}

func lookupUnaryElementwiseKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s unary kernel for %s", storageDType.Name(), name)
	}

	return kernel
}

func binaryElementwiseDTypeBytes(
	elementCount int,
	name string,
	storageDType dtype.DType,
) ([]byte, []byte, []byte) {
	leftValues, rightValues, _ := binaryFloat32ParityValues(elementCount, name)
	leftBytes := encodeFloat32ValuesAsDType(leftValues, storageDType)
	rightBytes := encodeFloat32ValuesAsDType(rightValues, storageDType)
	leftStored := decodeDTypeBytesToFloat32(leftBytes, storageDType)
	rightStored := decodeDTypeBytesToFloat32(rightBytes, storageDType)
	expectedValues := make([]float32, elementCount)

	for index := range expectedValues {
		expectedValues[index] = binaryFloat32Expected(name, leftStored[index], rightStored[index])
	}

	return leftBytes, rightBytes, encodeFloat32ValuesAsDType(expectedValues, storageDType)
}

func unaryElementwiseDTypeBytes(
	elementCount int,
	name string,
	storageDType dtype.DType,
) ([]byte, []byte) {
	inputValues, _ := unaryFloat32ParityValues(elementCount, name)
	inputBytes := encodeFloat32ValuesAsDType(inputValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expectedValues := make([]float32, elementCount)

	for index := range expectedValues {
		expectedValues[index] = unaryFloat32Expected(name, inputStored[index])
	}

	return inputBytes, encodeFloat32ValuesAsDType(expectedValues, storageDType)
}

func encodeFloat32ValuesAsDType(values []float32, storageDType dtype.DType) []byte {
	switch storageDType {
	case dtype.Float16:
		out := make([]dtype.F16, len(values))
		for index, value := range values {
			out[index] = dtype.Fromfloat32(value)
		}

		return dtypeconvert.Float16ToBytes(out)
	case dtype.BFloat16:
		out := make([]dtype.BF16, len(values))
		for index, value := range values {
			out[index] = dtype.NewBfloat16FromFloat32(value)
		}

		return dtypeconvert.BFloat16ToBytes(out)
	}

	panic("unsupported elementwise dtype: " + storageDType.Name())
}

func decodeDTypeBytesToFloat32(bytes []byte, storageDType dtype.DType) []float32 {
	values, err := dtypeconvert.BytesToFloat32(storageDType, bytes)
	if err != nil {
		panic(err)
	}

	return values
}

func uploadDTypeTensorForTest(
	testingObject testing.TB,
	backend *Backend,
	shape tensor.Shape,
	storageDType dtype.DType,
	bytes []byte,
) tensor.Tensor {
	testingObject.Helper()

	input, err := backend.Upload(shape, storageDType, bytes)
	if err != nil {
		testingObject.Fatal(err)
	}

	return input
}

func assertDTypeBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
	maxULP uint32,
) {
	testingObject.Helper()

	actualDType, actualBytes, err := backend.Download(input)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	if actualDType != storageDType {
		testingObject.Fatalf("download dtype mismatch: got %s want %s", actualDType, storageDType)
	}

	assertDTypeBytesWithinULP(testingObject, actualBytes, expectedBytes, maxULP)
}

func assertDTypeBytesWithinULP(
	testingObject testing.TB,
	actualBytes []byte,
	expectedBytes []byte,
	maxULP uint32,
) {
	testingObject.Helper()

	if len(actualBytes) != len(expectedBytes) {
		testingObject.Fatalf("byte length mismatch: got %d want %d", len(actualBytes), len(expectedBytes))
	}

	for index := 0; index < len(actualBytes); index += 2 {
		actualBits := binary.LittleEndian.Uint16(actualBytes[index:])
		expectedBits := binary.LittleEndian.Uint16(expectedBytes[index:])
		distance := uint16Distance(actualBits, expectedBits)

		if distance <= maxULP {
			continue
		}

		testingObject.Fatalf(
			"dtype bit mismatch at element %d: got %04x, want %04x, distance %d > %d",
			index/2,
			actualBits,
			expectedBits,
			distance,
			maxULP,
		)
	}
}

func uint16Distance(actual uint16, expected uint16) uint32 {
	actualOrdered := orderedUint16FloatBits(actual)
	expectedOrdered := orderedUint16FloatBits(expected)

	if actualOrdered > expectedOrdered {
		return uint32(actualOrdered - expectedOrdered)
	}

	return uint32(expectedOrdered - actualOrdered)
}

func orderedUint16FloatBits(bits uint16) int32 {
	signedBits := int32(int16(bits))
	if signedBits < 0 {
		return -32768 - signedBits
	}

	return signedBits
}
