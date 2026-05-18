package metal

import (
	"encoding/binary"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalEmbeddingDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalTransformerDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalEmbeddingDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalEmbeddingDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, hidden := range parityElementCounts {
		hidden := hidden

		testingObject.Run(testNameForElementCount(hidden), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" embedding tensors", testingObject, func() {
				runEmbeddingLookupParityCase(testingObject, backend, storageDType, hidden)
				runEmbeddingBagParityCase(testingObject, backend, storageDType, hidden)
			})
		})
	}
}

func runEmbeddingLookupParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	hidden int,
) {
	vocab, indexCount := 67, 13
	tableBytes, indicesBytes, expectedBytes := embeddingLookupDTypeBytes(
		vocab, hidden, indexCount, storageDType,
	)
	table, indices, out := embeddingLookupTensorsForTest(
		testingObject, backend, vocab, hidden, indexCount, storageDType,
		tableBytes, indicesBytes,
	)
	defer closeBenchmarkTensors(table, indices, out)

	err := lookupEmbeddingLookupKernel(testingObject, storageDType).Run(table, indices, out)
	convey.So(err, convey.ShouldBeNil)
	assertRawOrDTypeBytesForTest(testingObject, backend, out, storageDType, expectedBytes, 0)
}

func runEmbeddingBagParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	hidden int,
) {
	vocab, indexCount, bagCount := 71, 29, 7
	tableBytes, indicesBytes, offsetsBytes, expectedBytes := embeddingBagDTypeBytes(
		vocab, hidden, indexCount, bagCount, storageDType,
	)
	table, indices, offsets, out := embeddingBagTensorsForTest(
		testingObject, backend, vocab, hidden, bagCount, storageDType,
		tableBytes, indicesBytes, offsetsBytes,
	)
	defer closeBenchmarkTensors(table, indices, offsets, out)

	err := lookupEmbeddingBagKernel(testingObject, storageDType).Run(table, indices, offsets, out)
	convey.So(err, convey.ShouldBeNil)
	assertRawOrDTypeBytesForTest(testingObject, backend, out, storageDType, expectedBytes, 1)
}

func lookupEmbeddingLookupKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("embedding_lookup", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, dtype.Int32},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s embedding_lookup kernel", storageDType.Name())
	}

	return kernel
}

func lookupEmbeddingBagKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("embedding_bag", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, dtype.Int32, dtype.Int32},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s embedding_bag kernel", storageDType.Name())
	}

	return kernel
}

func embeddingLookupTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	vocab int,
	hidden int,
	indexCount int,
	storageDType dtype.DType,
	tableBytes []byte,
	indicesBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	table := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{vocab, hidden}),
		storageDType, tableBytes,
	)
	indices := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{indexCount}),
		dtype.Int32, indicesBytes,
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{indexCount, hidden}),
		storageDType,
	)

	return table, indices, out
}

func embeddingBagTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	vocab int,
	hidden int,
	bagCount int,
	storageDType dtype.DType,
	tableBytes []byte,
	indicesBytes []byte,
	offsetsBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	table := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{vocab, hidden}),
		storageDType, tableBytes,
	)
	indices := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{len(indicesBytes) / 4}),
		dtype.Int32, indicesBytes,
	)
	offsets := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{bagCount}),
		dtype.Int32, offsetsBytes,
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{bagCount, hidden}),
		storageDType,
	)

	return table, indices, offsets, out
}

func embeddingLookupDTypeBytes(
	vocab int,
	hidden int,
	indexCount int,
	storageDType dtype.DType,
) ([]byte, []byte, []byte) {
	tableValues := projectionValues(vocab*hidden, 43, 64)
	indices := embeddingIndices(indexCount, vocab)
	tableBytes := encodeProjectionValuesAsDType(tableValues, storageDType)
	tableStored := decodeDTypeBytesToFloat32(tableBytes, storageDType)
	expectedValues := embeddingLookupExpected(tableStored, indices, hidden)

	return tableBytes, int32ValuesToBytes(indices), encodeProjectionValuesAsDType(
		expectedValues, storageDType,
	)
}

func embeddingBagDTypeBytes(
	vocab int,
	hidden int,
	indexCount int,
	bagCount int,
	storageDType dtype.DType,
) ([]byte, []byte, []byte, []byte) {
	tableValues := projectionValues(vocab*hidden, 47, 128)
	indices := embeddingIndices(indexCount, vocab)
	offsets := embeddingOffsets(indexCount, bagCount)
	tableBytes := encodeProjectionValuesAsDType(tableValues, storageDType)
	tableStored := decodeDTypeBytesToFloat32(tableBytes, storageDType)
	expectedValues := embeddingBagExpected(tableStored, indices, offsets, hidden)

	return tableBytes, int32ValuesToBytes(indices), int32ValuesToBytes(offsets),
		encodeProjectionValuesAsDType(expectedValues, storageDType)
}

func embeddingIndices(indexCount int, vocab int) []int32 {
	indices := make([]int32, indexCount)

	for index := range indices {
		indices[index] = int32((index*7 + 3) % vocab)
	}

	return indices
}

func embeddingOffsets(indexCount int, bagCount int) []int32 {
	offsets := make([]int32, bagCount)

	for bagIndex := range offsets {
		offsets[bagIndex] = int32(bagIndex * indexCount / bagCount)
	}

	return offsets
}

func embeddingLookupExpected(table []float32, indices []int32, hidden int) []float32 {
	out := make([]float32, len(indices)*hidden)

	for resultIndex, tokenID := range indices {
		copy(
			out[resultIndex*hidden:(resultIndex+1)*hidden],
			table[int(tokenID)*hidden:(int(tokenID)+1)*hidden],
		)
	}

	return out
}

func embeddingBagExpected(
	table []float32,
	indices []int32,
	offsets []int32,
	hidden int,
) []float32 {
	out := make([]float32, len(offsets)*hidden)

	for bagIndex := range offsets {
		start := int(offsets[bagIndex])
		end := len(indices)
		if bagIndex+1 < len(offsets) {
			end = int(offsets[bagIndex+1])
		}

		for elementIndex := start; elementIndex < end; elementIndex++ {
			tokenID := int(indices[elementIndex])

			for hiddenIndex := range hidden {
				out[bagIndex*hidden+hiddenIndex] += table[tokenID*hidden+hiddenIndex]
			}
		}
	}

	return out
}

func int32ValuesToBytes(values []int32) []byte {
	bytes := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(bytes[index*4:], uint32(value))
	}

	return bytes
}

func assertRawOrDTypeBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
	maxULP uint32,
) {
	testingObject.Helper()

	if storageDType == dtype.Float32 {
		assertRawBytesForTest(testingObject, backend, input, storageDType, expectedBytes)
		return
	}

	assertDTypeBytesForTest(testingObject, backend, input, storageDType, expectedBytes, maxULP)
}
