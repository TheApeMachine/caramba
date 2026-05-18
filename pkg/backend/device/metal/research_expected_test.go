package metal

import "github.com/theapemachine/caramba/pkg/dtype"

type vsaBinaryFixture struct {
	leftBytes      []byte
	rightBytes     []byte
	expectedValues []float32
}

type vsaUnaryFixture struct {
	inputBytes     []byte
	expectedValues []float32
}

type pcPredictionErrorFixture struct {
	observedBytes  []byte
	predictedBytes []byte
	expectedValues []float32
}

type pcFixture struct {
	weightBytes                 []byte
	stateBytes                  []byte
	errorBytes                  []byte
	predictionValues            []float32
	updatedRepresentationValues []float32
	updatedWeightValues         []float32
}

func vsaBinaryFixtureForTest(
	name string,
	storageDType dtype.DType,
	elementCount int,
) vsaBinaryFixture {
	leftValues, rightValues := researchPairValues(elementCount)
	leftBytes := encodeResearchValuesAsDType(leftValues, storageDType)
	rightBytes := encodeResearchValuesAsDType(rightValues, storageDType)
	leftStored := decodeDTypeBytesToFloat32(leftBytes, storageDType)
	rightStored := decodeDTypeBytesToFloat32(rightBytes, storageDType)
	expectedValues := make([]float32, elementCount)

	for index, value := range leftStored {
		expectedValues[index] = researchBinaryExpected(name, value, rightStored[index])
	}

	return vsaBinaryFixture{
		leftBytes: leftBytes, rightBytes: rightBytes, expectedValues: expectedValues,
	}
}

func vsaUnaryFixtureForTest(
	name string,
	storageDType dtype.DType,
	elementCount int,
) vsaUnaryFixture {
	inputValues := researchSingleValues(elementCount)
	inputBytes := encodeResearchValuesAsDType(inputValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expectedValues := researchVSAUnaryExpected(name, inputStored)

	return vsaUnaryFixture{inputBytes: inputBytes, expectedValues: expectedValues}
}

func pcPredictionErrorFixtureForTest(
	storageDType dtype.DType,
	elementCount int,
) pcPredictionErrorFixture {
	observedValues, predictedValues := researchPairValues(elementCount)
	observedBytes := encodeResearchValuesAsDType(observedValues, storageDType)
	predictedBytes := encodeResearchValuesAsDType(predictedValues, storageDType)
	observedStored := decodeDTypeBytesToFloat32(observedBytes, storageDType)
	predictedStored := decodeDTypeBytesToFloat32(predictedBytes, storageDType)
	expectedValues := make([]float32, elementCount)

	for index, value := range observedStored {
		expectedValues[index] = value - predictedStored[index]
	}

	return pcPredictionErrorFixture{
		observedBytes:  observedBytes,
		predictedBytes: predictedBytes,
		expectedValues: expectedValues,
	}
}

func pcFixtureForTest(storageDType dtype.DType, outCount int, inCount int) pcFixture {
	weightValues := researchMatrixValues(outCount, inCount)
	stateValues := researchSingleValues(inCount)
	errorValues := researchSingleValues(outCount)
	weightBytes := encodeResearchValuesAsDType(weightValues, storageDType)
	stateBytes := encodeResearchValuesAsDType(stateValues, storageDType)
	errorBytes := encodeResearchValuesAsDType(errorValues, storageDType)
	weightStored := decodeDTypeBytesToFloat32(weightBytes, storageDType)
	stateStored := decodeDTypeBytesToFloat32(stateBytes, storageDType)
	errorStored := decodeDTypeBytesToFloat32(errorBytes, storageDType)

	return pcFixture{
		weightBytes:                 weightBytes,
		stateBytes:                  stateBytes,
		errorBytes:                  errorBytes,
		predictionValues:            pcPredictionExpected(weightStored, stateStored, outCount, inCount),
		updatedRepresentationValues: pcUpdateRepresentationExpected(weightStored, stateStored, errorStored, outCount, inCount),
		updatedWeightValues:         pcUpdateWeightsExpected(weightStored, stateStored, errorStored, outCount, inCount),
	}
}

func researchBinaryExpected(name string, left float32, right float32) float32 {
	if name == "vsa_bind" {
		return left * right
	}

	return left + right
}

func researchVSAUnaryExpected(name string, input []float32) []float32 {
	out := make([]float32, len(input))

	for index, value := range input {
		target := index + 1
		if name == "vsa_inverse_permute" {
			target = index - 1
		}

		if target == len(input) {
			target = 0
		}

		if target < 0 {
			target = len(input) - 1
		}

		out[target] = value
	}

	return out
}

func pcPredictionExpected(
	weights []float32,
	state []float32,
	outCount int,
	inCount int,
) []float32 {
	out := make([]float32, outCount)

	for outIndex := range outCount {
		var sum float32

		for inIndex := range inCount {
			sum += weights[outIndex*inCount+inIndex] * state[inIndex]
		}

		out[outIndex] = sum
	}

	return out
}

func pcUpdateRepresentationExpected(
	weights []float32,
	state []float32,
	predictionError []float32,
	outCount int,
	inCount int,
) []float32 {
	out := append([]float32(nil), state...)

	for outIndex := range outCount {
		for inIndex := range inCount {
			out[inIndex] += predictiveCodingLearningRateForTest *
				weights[outIndex*inCount+inIndex] *
				predictionError[outIndex]
		}
	}

	return out
}

func pcUpdateWeightsExpected(
	weights []float32,
	state []float32,
	predictionError []float32,
	outCount int,
	inCount int,
) []float32 {
	out := append([]float32(nil), weights...)

	for outIndex := range outCount {
		for inIndex := range inCount {
			index := outIndex*inCount + inIndex
			out[index] += predictiveCodingLearningRateForTest *
				predictionError[outIndex] *
				state[inIndex]
		}
	}

	return out
}

const predictiveCodingLearningRateForTest = 1.0e-2

func researchPairValues(elementCount int) ([]float32, []float32) {
	leftValues := make([]float32, elementCount)
	rightValues := make([]float32, elementCount)

	for index := range leftValues {
		leftValues[index] = centeredPowerOfTwoValue(index*7+3, 41, 16)
		rightValues[index] = centeredPowerOfTwoValue(index*13+5, 37, 19)
	}

	return leftValues, rightValues
}

func researchSingleValues(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*11+7, 43, 23)
	}

	return values
}

func researchMatrixValues(rows int, cols int) []float32 {
	values := make([]float32, rows*cols)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*17+9, 47, 29)
	}

	return values
}
