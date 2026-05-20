package metal

import (
	"encoding/binary"
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

type checkpointFixture struct {
	values       []float32
	inputBytes   []byte
	encodedBytes []byte
}

type tokenizerFixture struct {
	inputBytes []byte
}

type weightFreezeFixture struct {
	maskBytes     []byte
	gradientBytes []byte
	expectedBytes []byte
}

func checkpointFixtureForTest(elementCount int) checkpointFixture {
	values := checkpointValuesForTest(elementCount)
	inputBytes := encodeLossValuesAsDType(values, dtype.Float32)
	encodedBytes := checkpointEncodedBytesForTest(values, []int{elementCount})

	return checkpointFixture{
		values:       values,
		inputBytes:   inputBytes,
		encodedBytes: encodedBytes,
	}
}

func checkpointValuesForTest(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*13+5, 97, 23)
	}

	return values
}

func checkpointEncodedBytesForTest(values []float32, dims []int) []byte {
	headerBytes := 16 + len(dims)*8
	out := make([]byte, headerBytes+len(values)*4)
	binary.LittleEndian.PutUint64(out[0:8], uint64(len(dims)))
	binary.LittleEndian.PutUint64(out[8:16], uint64(len(values)*4))

	for index, dim := range dims {
		binary.LittleEndian.PutUint64(out[16+index*8:], uint64(dim))
	}

	for index, value := range values {
		binary.LittleEndian.PutUint32(out[headerBytes+index*4:], math.Float32bits(value))
	}

	return out
}

func tokenizerFixtureForTest(elementCount int) tokenizerFixture {
	values := make([]int32, elementCount)
	out := make([]byte, elementCount*4)

	for index := range values {
		values[index] = int32((index*7919)%65521) - 32768
		binary.LittleEndian.PutUint32(out[index*4:], uint32(values[index]))
	}

	return tokenizerFixture{inputBytes: out}
}

func weightFreezeFixtureForTest(
	elementCount int,
	storageDType dtype.DType,
) weightFreezeFixture {
	values := weightFreezeValuesForTest(elementCount)
	gradientBytes := encodeLossValuesAsDType(values, storageDType)
	storedValues := decodeDTypeBytesToFloat32(gradientBytes, storageDType)
	maskBytes := weightFreezeMaskBytesForTest(elementCount)
	expected := weightFreezeExpectedForTest(storedValues)

	return weightFreezeFixture{
		maskBytes:     maskBytes,
		gradientBytes: gradientBytes,
		expectedBytes: encodeLossValuesAsDType(expected, storageDType),
	}
}

func weightFreezeValuesForTest(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*19+7, 83, 29)
	}

	return values
}

func weightFreezeMaskBytesForTest(elementCount int) []byte {
	out := make([]byte, (elementCount+7)/8)

	for index := 0; index < elementCount; index++ {
		if weightFreezeMaskBitForTest(index) {
			out[index>>3] |= 1 << (index & 7)
		}
	}

	return out
}

func weightFreezeMaskBitForTest(index int) bool {
	return index%3 != 1 && index%11 != 7
}

func weightFreezeExpectedForTest(input []float32) []float32 {
	out := make([]float32, len(input))

	for index, value := range input {
		if weightFreezeMaskBitForTest(index) {
			out[index] = value
		}
	}

	return out
}
