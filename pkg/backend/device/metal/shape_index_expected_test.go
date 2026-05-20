package metal

import "github.com/theapemachine/manifesto/dtype"

type shapeIndexFixture struct {
	firstBytes     []byte
	secondBytes    []byte
	thirdBytes     []byte
	permutation    []int32
	expectedBytes  []byte
	expectedValues []float32
}

func gatherFixtureForTest(
	outRows int,
	inner int,
	storageDType dtype.DType,
) shapeIndexFixture {
	sourceRows := outRows + 5
	sourceBytes := encodeProjectionValuesAsDType(
		shapeIndexValues(sourceRows*inner, 17), storageDType,
	)
	sourceValues := decodeDTypeBytesToFloat32(sourceBytes, storageDType)
	indices := gatherIndices(outRows, sourceRows)
	expected := gatherExpected(sourceValues, indices, inner)

	return shapeIndexFixture{
		firstBytes:     sourceBytes,
		secondBytes:    int32ValuesToBytes(indices),
		expectedBytes:  encodeProjectionValuesAsDType(expected, storageDType),
		expectedValues: expected,
	}
}

func scatterFixtureForTest(
	updateRows int,
	inner int,
	storageDType dtype.DType,
) shapeIndexFixture {
	targetRows := updateRows + 5
	targetBytes := encodeProjectionValuesAsDType(
		shapeIndexValues(targetRows*inner, 23), storageDType,
	)
	updateBytes := encodeProjectionValuesAsDType(
		shapeIndexValues(updateRows*inner, 29), storageDType,
	)
	targetValues := decodeDTypeBytesToFloat32(targetBytes, storageDType)
	updateValues := decodeDTypeBytesToFloat32(updateBytes, storageDType)
	indices := scatterIndices(updateRows, targetRows)
	expected := scatterExpected(targetValues, indices, updateValues, inner)

	return shapeIndexFixture{
		firstBytes:     targetBytes,
		secondBytes:    int32ValuesToBytes(indices),
		thirdBytes:     updateBytes,
		expectedBytes:  encodeProjectionValuesAsDType(expected, storageDType),
		expectedValues: expected,
	}
}

func whereFixtureForTest(elementCount int, storageDType dtype.DType) shapeIndexFixture {
	positiveBytes := encodeProjectionValuesAsDType(shapeIndexValues(elementCount, 31), storageDType)
	negativeBytes := encodeProjectionValuesAsDType(shapeIndexValues(elementCount, 37), storageDType)
	positive := decodeDTypeBytesToFloat32(positiveBytes, storageDType)
	negative := decodeDTypeBytesToFloat32(negativeBytes, storageDType)
	maskBytes := shapeMaskBytes(elementCount)
	expected := whereExpected(positive, negative)

	return shapeIndexFixture{
		firstBytes:     maskBytes,
		secondBytes:    positiveBytes,
		thirdBytes:     negativeBytes,
		expectedBytes:  encodeProjectionValuesAsDType(expected, storageDType),
		expectedValues: expected,
	}
}

func maskedFillFixtureForTest(elementCount int, storageDType dtype.DType) shapeIndexFixture {
	inputBytes := encodeProjectionValuesAsDType(shapeIndexValues(elementCount, 41), storageDType)
	scalarBytes := encodeProjectionValuesAsDType([]float32{-0.3125}, storageDType)
	input := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	scalar := decodeDTypeBytesToFloat32(scalarBytes, storageDType)[0]
	maskBytes := shapeMaskBytes(elementCount)
	expected := maskedFillExpected(input, scalar)

	return shapeIndexFixture{
		firstBytes:     inputBytes,
		secondBytes:    maskBytes,
		thirdBytes:     scalarBytes,
		expectedBytes:  encodeProjectionValuesAsDType(expected, storageDType),
		expectedValues: expected,
	}
}

func transposeFixtureForTest(elementCount int, storageDType dtype.DType) shapeIndexFixture {
	dims := transposeInputDims(elementCount)
	values := shapeIndexValues(dims[0]*dims[1]*dims[2], 43)
	inputBytes := encodeProjectionValuesAsDType(values, storageDType)
	input := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	permutation := []int32{2, 1, 0}
	expected := transposeExpected(input, dims, permutation)

	return shapeIndexFixture{
		firstBytes:     inputBytes,
		permutation:    permutation,
		secondBytes:    int32ValuesToBytes(permutation),
		expectedBytes:  encodeProjectionValuesAsDType(expected, storageDType),
		expectedValues: expected,
	}
}

func shapeIndexValues(count int, salt int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*salt+5, 97, 23)
	}

	return values
}

func gatherIndices(outRows int, sourceRows int) []int32 {
	indices := make([]int32, outRows)

	for index := range indices {
		indices[index] = int32((index*7 + 3) % sourceRows)
	}

	return indices
}

func scatterIndices(updateRows int, targetRows int) []int32 {
	indices := make([]int32, updateRows)

	for index := range indices {
		indices[index] = int32((index*5 + 2) % targetRows)
	}

	return indices
}

func gatherExpected(source []float32, indices []int32, inner int) []float32 {
	out := make([]float32, len(indices)*inner)

	for outRow, sourceRow := range indices {
		sourceOffset := int(sourceRow) * inner
		copy(out[outRow*inner:(outRow+1)*inner], source[sourceOffset:sourceOffset+inner])
	}

	return out
}

func scatterExpected(target []float32, indices []int32, updates []float32, inner int) []float32 {
	out := append([]float32(nil), target...)

	for updateRow, targetRow := range indices {
		targetOffset := int(targetRow) * inner
		updateOffset := updateRow * inner
		copy(out[targetOffset:targetOffset+inner], updates[updateOffset:updateOffset+inner])
	}

	return out
}

func shapeMaskBytes(elementCount int) []byte {
	out := make([]byte, (elementCount+7)/8)

	for index := range elementCount {
		if shapeMaskBit(index) {
			out[index>>3] |= 1 << (index & 7)
		}
	}

	return out
}

func shapeMaskBit(index int) bool {
	return index%3 == 0 || index%11 == 5
}

func whereExpected(positive []float32, negative []float32) []float32 {
	out := make([]float32, len(positive))

	for index := range out {
		out[index] = negative[index]

		if shapeMaskBit(index) {
			out[index] = positive[index]
		}
	}

	return out
}

func maskedFillExpected(input []float32, scalar float32) []float32 {
	out := append([]float32(nil), input...)

	for index := range out {
		if shapeMaskBit(index) {
			out[index] = scalar
		}
	}

	return out
}

func transposeInputDims(elementCount int) []int {
	return []int{2, elementCount, 3}
}

func transposeOutputDims(elementCount int) []int {
	return []int{3, elementCount, 2}
}

func transposeExpected(input []float32, dims []int, permutation []int32) []float32 {
	outDims := transposeOutputDims(dims[1])
	out := make([]float32, len(input))

	for firstIndex := range dims[0] {
		for secondIndex := range dims[1] {
			for thirdIndex := range dims[2] {
				inputCoords := []int{firstIndex, secondIndex, thirdIndex}
				outCoords := []int{
					inputCoords[int(permutation[0])],
					inputCoords[int(permutation[1])],
					inputCoords[int(permutation[2])],
				}
				inputIndex := (firstIndex*dims[1]+secondIndex)*dims[2] + thirdIndex
				outIndex := (outCoords[0]*outDims[1]+outCoords[1])*outDims[2] + outCoords[2]
				out[outIndex] = input[inputIndex]
			}
		}
	}

	return out
}
