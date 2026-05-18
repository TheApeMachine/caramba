package metal

import (
	"math"

	"github.com/theapemachine/caramba/pkg/dtype"
)

type mathUnaryFixture struct {
	inputBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

type mathOuterFixture struct {
	leftBytes       []byte
	rightBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

func invSqrtDimScaleFixtureForTest(
	elementCount int,
	storageDType dtype.DType,
	scaleDim int32,
) mathUnaryFixture {
	inputValues := mathInputValues(elementCount)
	inputBytes := encodeLossValuesAsDType(inputValues, storageDType)
	storedInput := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expected := invSqrtDimScaleExpected(storedInput, scaleDim)

	return mathUnaryFixture{
		inputBytes:      inputBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func logSumExpFixtureForTest(
	rows int,
	cols int,
	storageDType dtype.DType,
) mathUnaryFixture {
	inputValues := mathInputValues(rows * cols)
	inputBytes := encodeLossValuesAsDType(inputValues, storageDType)
	storedInput := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expected := logSumExpExpected(storedInput, rows, cols)

	return mathUnaryFixture{
		inputBytes:      inputBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func outerFixtureForTest(rows int, cols int, storageDType dtype.DType) mathOuterFixture {
	leftValues := mathInputValues(rows)
	rightValues := mathInputValues(cols)
	leftBytes := encodeLossValuesAsDType(leftValues, storageDType)
	rightBytes := encodeLossValuesAsDType(rightValues, storageDType)
	leftStored := decodeDTypeBytesToFloat32(leftBytes, storageDType)
	rightStored := decodeDTypeBytesToFloat32(rightBytes, storageDType)
	expected := outerExpected(leftStored, rightStored)

	return mathOuterFixture{
		leftBytes:       leftBytes,
		rightBytes:      rightBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func mathInputValues(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*19+7, 71, 23)
	}

	return values
}

func invSqrtDimScaleExpected(input []float32, scaleDim int32) []float32 {
	out := make([]float32, len(input))
	scale := float32(1.0 / math.Sqrt(float64(scaleDim)))

	for index, value := range input {
		out[index] = value * scale
	}

	return out
}

func logSumExpExpected(input []float32, rows int, cols int) []float32 {
	out := make([]float32, rows)

	for rowIndex := range rows {
		rowOffset := rowIndex * cols
		maximum := logSumExpRowMaximum(input, rowOffset, cols)
		sum := logSumExpRowSum(input, rowOffset, cols, maximum)
		out[rowIndex] = maximum + float32(math.Log(float64(sum)))
	}

	return out
}

func logSumExpRowMaximum(input []float32, rowOffset int, cols int) float32 {
	reduction := make([]float32, metalReductionThreadCount)

	for threadIndex := range metalReductionThreadCount {
		reduction[threadIndex] = -math.MaxFloat32
		for col := threadIndex; col < cols; col += metalReductionThreadCount {
			reduction[threadIndex] = max(reduction[threadIndex], input[rowOffset+col])
		}
	}

	return lossReduceMaxFloat32(reduction)
}

func logSumExpRowSum(input []float32, rowOffset int, cols int, maximum float32) float32 {
	reduction := make([]float32, metalReductionThreadCount)

	for threadIndex := range metalReductionThreadCount {
		for col := threadIndex; col < cols; col += metalReductionThreadCount {
			reduction[threadIndex] += float32(math.Exp(float64(input[rowOffset+col] - maximum)))
		}
	}

	return lossReduceFloat32(reduction)
}

func outerExpected(left []float32, right []float32) []float32 {
	out := make([]float32, len(left)*len(right))

	for leftIndex, leftValue := range left {
		for rightIndex, rightValue := range right {
			out[leftIndex*len(right)+rightIndex] = leftValue * rightValue
		}
	}

	return out
}
