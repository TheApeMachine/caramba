package metal

import (
	"encoding/binary"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func lookupUnaryShapeKernel(
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
		testingObject.Fatalf("missing Metal %s unary shape kernel for %s", storageDType.Name(), name)
	}

	return kernel
}

func lookupBinaryShapeKernel(
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
		testingObject.Fatalf("missing Metal %s binary shape kernel for %s", storageDType.Name(), name)
	}

	return kernel
}

func lookupSplit2Kernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("split2", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType, storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s split2 kernel", storageDType.Name())
	}

	return kernel
}

func lookupViewAsHeadsKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("view_as_heads", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, dtype.Int32},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s view_as_heads kernel", storageDType.Name())
	}

	return kernel
}

func mustShapeForTest(testingObject testing.TB, dims []int) tensor.Shape {
	testingObject.Helper()

	shape, err := tensor.NewShape(dims)
	if err != nil {
		testingObject.Fatal(err)
	}

	return shape
}

func emptyTensorForTest(
	testingObject testing.TB,
	backend *Backend,
	shape tensor.Shape,
	storageDType dtype.DType,
) tensor.Tensor {
	testingObject.Helper()

	out, err := backend.bridge.empty(shape, storageDType)
	if err != nil {
		testingObject.Fatal(err)
	}

	return out
}

func rawShapeBytesForTest(
	testingObject testing.TB,
	shape tensor.Shape,
	storageDType dtype.DType,
) []byte {
	testingObject.Helper()

	byteCount, err := shape.Bytes(storageDType)
	if err != nil {
		testingObject.Fatal(err)
	}

	return shiftedRawByteCountForTest(byteCount, 0)
}

func shiftedRawBytesForTest(
	testingObject testing.TB,
	shape tensor.Shape,
	storageDType dtype.DType,
	shift int,
) []byte {
	testingObject.Helper()

	byteCount, err := shape.Bytes(storageDType)
	if err != nil {
		testingObject.Fatal(err)
	}

	return shiftedRawByteCountForTest(byteCount, shift)
}

func shiftedRawByteCountForTest(byteCount int, shift int) []byte {
	bytes := make([]byte, byteCount)

	for index := range bytes {
		bytes[index] = byte((index*31 + shift + 7) % 251)
	}

	return bytes
}

func splitRawBytesForTest(inputBytes []byte) ([]byte, []byte) {
	midpoint := len(inputBytes) / 2
	return append([]byte(nil), inputBytes[:midpoint]...), append([]byte(nil), inputBytes[midpoint:]...)
}

func lastTokenRawBytesForTest(
	testingObject testing.TB,
	inputBytes []byte,
	storageDType dtype.DType,
	batch int,
	seq int,
	hidden int,
) []byte {
	testingObject.Helper()

	elementBytes := dtypeSizeForTest(testingObject, storageDType)
	hiddenBytes := hidden * elementBytes
	out := make([]byte, batch*hiddenBytes)

	for batchIndex := range batch {
		inputOffset := (batchIndex*seq + seq - 1) * hiddenBytes
		outOffset := batchIndex * hiddenBytes
		copy(out[outOffset:outOffset+hiddenBytes], inputBytes[inputOffset:inputOffset+hiddenBytes])
	}

	return out
}

func transpose2DRawBytesForTest(
	testingObject testing.TB,
	inputBytes []byte,
	storageDType dtype.DType,
	rows int,
	cols int,
) []byte {
	testingObject.Helper()

	elementBytes := dtypeSizeForTest(testingObject, storageDType)
	out := make([]byte, len(inputBytes))

	for row := range rows {
		for col := range cols {
			inputOffset := (row*cols + col) * elementBytes
			outOffset := (col*rows + row) * elementBytes
			copy(out[outOffset:outOffset+elementBytes], inputBytes[inputOffset:inputOffset+elementBytes])
		}
	}

	return out
}

func upsampleNearest2DRawBytesForTest(
	testingObject testing.TB,
	inputBytes []byte,
	storageDType dtype.DType,
	inputDims []int,
	outDims []int,
) []byte {
	testingObject.Helper()

	elementBytes := dtypeSizeForTest(testingObject, storageDType)
	outElements := outDims[0] * outDims[1] * outDims[2] * outDims[3]
	out := make([]byte, outElements*elementBytes)

	for outIndex := range outElements {
		outCol := outIndex % outDims[3]
		outRow := (outIndex / outDims[3]) % outDims[2]
		channel := (outIndex / (outDims[3] * outDims[2])) % outDims[1]
		batch := outIndex / (outDims[3] * outDims[2] * outDims[1])
		inRow := outRow * inputDims[2] / outDims[2]
		inCol := outCol * inputDims[3] / outDims[3]
		inputIndex := ((batch*inputDims[1]+channel)*inputDims[2]+inRow)*inputDims[3] + inCol
		copyElementBytes(out, inputBytes, outIndex, inputIndex, elementBytes)
	}

	return out
}

func copyElementBytes(
	out []byte,
	input []byte,
	outIndex int,
	inputIndex int,
	elementBytes int,
) {
	outOffset := outIndex * elementBytes
	inputOffset := inputIndex * elementBytes
	copy(out[outOffset:outOffset+elementBytes], input[inputOffset:inputOffset+elementBytes])
}

func dtypeSizeForTest(testingObject testing.TB, storageDType dtype.DType) int {
	testingObject.Helper()

	size, err := storageDType.Size()
	if err != nil {
		testingObject.Fatal(err)
	}

	return size
}

func uploadInt32ScalarForTest(
	testingObject testing.TB,
	backend *Backend,
	value int32,
) tensor.Tensor {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{1})
	bytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(bytes, uint32(value))
	return uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Int32, bytes)
}

func assertRawBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
) {
	testingObject.Helper()

	actualDType, actualBytes, err := backend.Download(input)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	if actualDType != storageDType {
		testingObject.Fatalf("download dtype mismatch: got %s want %s", actualDType, storageDType)
	}

	if len(actualBytes) != len(expectedBytes) {
		testingObject.Fatalf("byte length mismatch: got %d want %d", len(actualBytes), len(expectedBytes))
	}

	for index := range actualBytes {
		if actualBytes[index] == expectedBytes[index] {
			continue
		}

		testingObject.Fatalf(
			"raw byte mismatch at %d: got %02x want %02x",
			index,
			actualBytes[index],
			expectedBytes[index],
		)
	}
}
