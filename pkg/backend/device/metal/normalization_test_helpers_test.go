package metal

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func encodeNormValuesAsDType(values []float32, storageDType dtype.DType) []byte {
	if storageDType == dtype.Float32 {
		return dtypeconvert.Float32ToBytes(values)
	}

	return encodeFloat32ValuesAsDType(values, storageDType)
}

func expectedLayerNormBytesForTest(
	rows int,
	cols int,
	storageDType dtype.DType,
) []byte {
	inputBytes, scaleBytes, biasBytes := normDTypeBytes(rows, cols, storageDType)
	inputValues := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	scaleValues := decodeDTypeBytesToFloat32(scaleBytes, storageDType)
	biasValues := decodeDTypeBytesToFloat32(biasBytes, storageDType)
	expectedValues := expectedLayerNormValuesForTest(inputValues, scaleValues, biasValues, rows, cols)
	return encodeNormValuesAsDType(expectedValues, storageDType)
}

func expectedRMSNormBytesForTest(
	rows int,
	cols int,
	storageDType dtype.DType,
) []byte {
	inputBytes, scaleBytes, _ := normDTypeBytes(rows, cols, storageDType)
	inputValues := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	scaleValues := decodeDTypeBytesToFloat32(scaleBytes, storageDType)
	expectedValues := expectedRMSNormValuesForTest(inputValues, scaleValues, rows, cols)
	return encodeNormValuesAsDType(expectedValues, storageDType)
}

func expectedLayerNormValuesForTest(
	input []float32,
	scale []float32,
	bias []float32,
	rows int,
	cols int,
) []float32 {
	out := make([]float32, len(input))

	for rowIndex := range rows {
		rowOffset := rowIndex * cols
		row := input[rowOffset : rowOffset+cols]
		outRow := out[rowOffset : rowOffset+cols]
		mean := normalizationMeanForTest(row)
		variance := normalizationVarianceForTest(row, mean)
		invStdDev := 1 / float32(math.Sqrt(float64(variance+layerNormEpsilonMetalForTest)))
		applyLayerNormExpectedRowForTest(row, outRow, scale, bias, mean, invStdDev)
	}

	return out
}

func expectedRMSNormValuesForTest(
	input []float32,
	scale []float32,
	rows int,
	cols int,
) []float32 {
	out := make([]float32, len(input))

	for rowIndex := range rows {
		rowOffset := rowIndex * cols
		row := input[rowOffset : rowOffset+cols]
		outRow := out[rowOffset : rowOffset+cols]
		meanSquare := normalizationMeanSquareForTest(row)
		invRMS := 1 / float32(math.Sqrt(float64(meanSquare+rmsNormEpsilonMetalForTest)))
		applyRMSNormExpectedRowForTest(row, outRow, scale, invRMS)
	}

	return out
}

const layerNormEpsilonMetalForTest = 1.0e-5
const rmsNormEpsilonMetalForTest = 1.0e-6

func normalizationMeanForTest(row []float32) float32 {
	var sum float32

	for _, value := range row {
		sum += value
	}

	return sum / float32(len(row))
}

func normalizationVarianceForTest(row []float32, mean float32) float32 {
	var variance float32

	for _, value := range row {
		delta := value - mean
		variance += delta * delta
	}

	return variance / float32(len(row))
}

func normalizationMeanSquareForTest(row []float32) float32 {
	var meanSquare float32

	for _, value := range row {
		meanSquare += value * value
	}

	return meanSquare / float32(len(row))
}

func applyLayerNormExpectedRowForTest(
	row []float32,
	outRow []float32,
	scale []float32,
	bias []float32,
	mean float32,
	invStdDev float32,
) {
	for index, value := range row {
		outRow[index] = (value-mean)*invStdDev*scale[index] + bias[index]
	}
}

func applyRMSNormExpectedRowForTest(
	row []float32,
	outRow []float32,
	scale []float32,
	invRMS float32,
) {
	for index, value := range row {
		outRow[index] = value * invRMS * scale[index]
	}
}

func assertNormalizationBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, expectedBytes, 2)
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
	assertNormalizationFloat32WithinULP(
		testingObject,
		actualValues,
		expectedValues,
		normalizationFloat32MaxULP,
	)
}

func assertNormalizationFloat32WithinULP(
	testingObject testing.TB,
	actualValues []float32,
	expectedValues []float32,
	maxULP uint32,
) {
	testingObject.Helper()

	maxDistance, maxIndex := maxNormalizationFloat32ULPDistance(actualValues, expectedValues)
	if maxDistance <= maxULP {
		return
	}

	testingObject.Fatalf(
		"normalization float32 max ULP mismatch at %d: got %08x (%g), want %08x (%g), distance %d > %d",
		maxIndex,
		math.Float32bits(actualValues[maxIndex]),
		actualValues[maxIndex],
		math.Float32bits(expectedValues[maxIndex]),
		expectedValues[maxIndex],
		maxDistance,
		maxULP,
	)
}

func maxNormalizationFloat32ULPDistance(
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
