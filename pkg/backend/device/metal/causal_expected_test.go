package metal

import (
	"math"

	"github.com/theapemachine/caramba/pkg/dtype"
)

type causalUnaryFixture struct {
	inputBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

type causalBinaryFixture struct {
	leftBytes       []byte
	rightBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

type causalTernaryFixture struct {
	firstBytes      []byte
	secondBytes     []byte
	thirdBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

type causalCounterfactualFixture struct {
	observedYBytes       []byte
	observedXBytes       []byte
	counterfactualXBytes []byte
	slopeBytes           []byte
	expectedBytes        []byte
	expectedFloat32      []float32
}

func backdoorFixtureForTest(
	xCount int,
	zCount int,
	yCount int,
	storageDType dtype.DType,
) causalBinaryFixture {
	conditionalBytes := encodeProjectionValuesAsDType(
		causalPositiveValues(xCount*zCount*yCount, 11), storageDType,
	)
	marginalBytes := encodeProjectionValuesAsDType(causalPositiveValues(zCount, 13), storageDType)
	conditional := decodeDTypeBytesToFloat32(conditionalBytes, storageDType)
	marginal := decodeDTypeBytesToFloat32(marginalBytes, storageDType)
	expected := backdoorExpected(conditional, marginal, xCount, zCount, yCount)

	return causalBinaryFixture{
		leftBytes: conditionalBytes, rightBytes: marginalBytes,
		expectedBytes: encodeProjectionValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func frontdoorFixtureForTest(
	xCount int,
	mCount int,
	yCount int,
	storageDType dtype.DType,
) causalTernaryFixture {
	mediatorBytes := encodeProjectionValuesAsDType(causalPositiveValues(xCount*mCount, 17), storageDType)
	outcomeBytes := encodeProjectionValuesAsDType(
		causalPositiveValues(xCount*mCount*yCount, 19), storageDType,
	)
	marginalBytes := encodeProjectionValuesAsDType(causalPositiveValues(xCount, 23), storageDType)
	mediator := decodeDTypeBytesToFloat32(mediatorBytes, storageDType)
	outcome := decodeDTypeBytesToFloat32(outcomeBytes, storageDType)
	marginal := decodeDTypeBytesToFloat32(marginalBytes, storageDType)
	expected := frontdoorExpected(mediator, outcome, marginal, xCount, mCount, yCount)

	return causalTernaryFixture{
		firstBytes: mediatorBytes, secondBytes: outcomeBytes, thirdBytes: marginalBytes,
		expectedBytes: encodeProjectionValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func doInterveneFixtureForTest(nodeCount int, storageDType dtype.DType) (causalBinaryFixture, []int32) {
	adjacencyBytes := encodeProjectionValuesAsDType(causalSignedValues(nodeCount*nodeCount, 29), storageDType)
	intervened := []int32{1, int32(nodeCount - 1), -1, int32(nodeCount + 3)}
	adjacency := decodeDTypeBytesToFloat32(adjacencyBytes, storageDType)
	expected := doInterveneExpected(adjacency, intervened, nodeCount)

	return causalBinaryFixture{
		leftBytes: adjacencyBytes, rightBytes: int32ValuesToBytes(intervened),
		expectedBytes: encodeProjectionValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}, intervened
}

func cateFixtureForTest(count int, storageDType dtype.DType) causalBinaryFixture {
	treatedBytes := encodeProjectionValuesAsDType(causalSignedValues(count, 31), storageDType)
	controlBytes := encodeProjectionValuesAsDType(causalSignedValues(count, 37), storageDType)
	treated := decodeDTypeBytesToFloat32(treatedBytes, storageDType)
	control := decodeDTypeBytesToFloat32(controlBytes, storageDType)
	expected := make([]float32, count)

	for index := range expected {
		expected[index] = treated[index] - control[index]
	}

	return causalBinaryFixture{
		leftBytes: treatedBytes, rightBytes: controlBytes,
		expectedBytes: encodeProjectionValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func counterfactualFixtureForTest(count int, storageDType dtype.DType) causalCounterfactualFixture {
	observedYBytes := encodeProjectionValuesAsDType(causalSignedValues(count, 41), storageDType)
	observedXBytes := encodeProjectionValuesAsDType(causalSignedValues(count, 43), storageDType)
	counterfactualXBytes := encodeProjectionValuesAsDType(causalSignedValues(count, 47), storageDType)
	slopeBytes := encodeProjectionValuesAsDType([]float32{0.375}, storageDType)
	observedY := decodeDTypeBytesToFloat32(observedYBytes, storageDType)
	observedX := decodeDTypeBytesToFloat32(observedXBytes, storageDType)
	counterfactualX := decodeDTypeBytesToFloat32(counterfactualXBytes, storageDType)
	slope := decodeDTypeBytesToFloat32(slopeBytes, storageDType)[0]
	expected := counterfactualExpected(observedY, observedX, counterfactualX, slope)

	return causalCounterfactualFixture{
		observedYBytes: observedYBytes, observedXBytes: observedXBytes,
		counterfactualXBytes: counterfactualXBytes, slopeBytes: slopeBytes,
		expectedBytes: encodeProjectionValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func ivFixtureForTest(count int, storageDType dtype.DType) causalTernaryFixture {
	instrumentBytes := encodeProjectionValuesAsDType(causalIVInstrumentValues(count), storageDType)
	treatmentBytes := encodeProjectionValuesAsDType(causalIVTreatmentValues(count), storageDType)
	outcomeBytes := encodeProjectionValuesAsDType(causalIVOutcomeValues(count), storageDType)
	instrument := decodeDTypeBytesToFloat32(instrumentBytes, storageDType)
	treatment := decodeDTypeBytesToFloat32(treatmentBytes, storageDType)
	outcome := decodeDTypeBytesToFloat32(outcomeBytes, storageDType)
	expected := []float32{ivExpected(instrument, treatment, outcome)}

	return causalTernaryFixture{
		firstBytes: instrumentBytes, secondBytes: treatmentBytes, thirdBytes: outcomeBytes,
		expectedBytes: encodeProjectionValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func dagFixtureForTest(count int, storageDType dtype.DType) (causalUnaryFixture, []int32) {
	conditionalsBytes := encodeProjectionValuesAsDType(causalDAGValues(count), storageDType)
	parents := make([]int32, count)
	conditionals := decodeDTypeBytesToFloat32(conditionalsBytes, storageDType)
	expected := []float32{dagExpected(conditionals)}

	return causalUnaryFixture{
		inputBytes:      conditionalsBytes,
		expectedBytes:   encodeProjectionValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}, parents
}

func backdoorExpected(
	conditional []float32,
	marginal []float32,
	xCount int,
	zCount int,
	yCount int,
) []float32 {
	out := make([]float32, xCount*yCount)

	for xIndex := range xCount {
		for yIndex := range yCount {
			var total float32

			for zIndex := range zCount {
				total += conditional[(xIndex*zCount+zIndex)*yCount+yIndex] * marginal[zIndex]
			}

			out[xIndex*yCount+yIndex] = total
		}
	}

	return out
}

func frontdoorExpected(
	mediator []float32,
	outcome []float32,
	marginal []float32,
	xCount int,
	mCount int,
	yCount int,
) []float32 {
	out := make([]float32, xCount*yCount)

	for xIndex := range xCount {
		for yIndex := range yCount {
			var total float32

			for mIndex := range mCount {
				var innerSum float32

				for xPrimeIndex := range xCount {
					innerSum += outcome[(xPrimeIndex*mCount+mIndex)*yCount+yIndex] * marginal[xPrimeIndex]
				}

				total += mediator[xIndex*mCount+mIndex] * innerSum
			}

			out[xIndex*yCount+yIndex] = total
		}
	}

	return out
}

func doInterveneExpected(adjacency []float32, intervened []int32, nodeCount int) []float32 {
	out := append([]float32(nil), adjacency...)

	for _, nodeID := range intervened {
		target := int(nodeID)
		if target < 0 || target >= nodeCount {
			continue
		}

		for sourceIndex := range nodeCount {
			out[sourceIndex*nodeCount+target] = 0
		}
	}

	return out
}

func counterfactualExpected(
	observedY []float32,
	observedX []float32,
	counterfactualX []float32,
	slope float32,
) []float32 {
	out := make([]float32, len(observedY))

	for index := range out {
		out[index] = observedY[index] + slope*(counterfactualX[index]-observedX[index])
	}

	return out
}

func causalPositiveValues(count int, salt int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = 0.125 + float32((index*salt+7)%19)/128
	}

	return values
}

func causalSignedValues(count int, salt int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = float32((index*salt+11)%41-20) / 32
	}

	return values
}

func causalIVInstrumentValues(count int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = float32(index%17) / 16
	}

	return values
}

func causalIVTreatmentValues(count int) []float32 {
	values := causalIVInstrumentValues(count)

	for index := range values {
		values[index] = 0.25 + 1.5*values[index] + float32(index%5)/64
	}

	return values
}

func causalIVOutcomeValues(count int) []float32 {
	values := causalIVInstrumentValues(count)

	for index := range values {
		values[index] = -0.125 + 0.75*values[index] + float32(index%7)/128
	}

	return values
}

func causalDAGValues(count int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = 0.875 + float32(index%5)/128
	}

	return values
}

func ivExpected(instrument []float32, treatment []float32, outcome []float32) float32 {
	sumZ, sumX, sumY, sumZY, sumZX := ivReductionTotals(instrument, treatment, outcome)
	count := float32(len(instrument))
	denominator := sumZX - (sumZ*sumX)/count
	numerator := sumZY - (sumZ*sumY)/count

	if float32(math.Abs(float64(denominator))) < 1.0e-12 {
		return 0
	}

	return numerator / denominator
}

func ivReductionTotals(
	instrument []float32,
	treatment []float32,
	outcome []float32,
) (float32, float32, float32, float32, float32) {
	partialCount := metalCausalPartialCount(len(instrument))
	scratch := make([][5]float32, partialCount)

	for groupIndex := range partialCount {
		scratch[groupIndex] = ivPartialForGroup(instrument, treatment, outcome, groupIndex)
	}

	return ivFinalizePartials(scratch)
}

func ivPartialForGroup(
	instrument []float32,
	treatment []float32,
	outcome []float32,
	groupIndex int,
) [5]float32 {
	values := [5][256]float32{}

	for threadIndex := range 256 {
		valueIndex := groupIndex*256 + threadIndex
		if valueIndex >= len(instrument) {
			continue
		}

		instrumentValue := instrument[valueIndex]
		treatmentValue := treatment[valueIndex]
		outcomeValue := outcome[valueIndex]
		values[0][threadIndex] = instrumentValue
		values[1][threadIndex] = treatmentValue
		values[2][threadIndex] = outcomeValue
		values[3][threadIndex] = instrumentValue * outcomeValue
		values[4][threadIndex] = instrumentValue * treatmentValue
	}

	return reduceFiveArrays(values)
}

func ivFinalizePartials(scratch [][5]float32) (float32, float32, float32, float32, float32) {
	values := [5][256]float32{}

	for threadIndex := range 256 {
		for partialIndex := threadIndex; partialIndex < len(scratch); partialIndex += 256 {
			for valueIndex := range 5 {
				values[valueIndex][threadIndex] += scratch[partialIndex][valueIndex]
			}
		}
	}

	reduced := reduceFiveArrays(values)
	return reduced[0], reduced[1], reduced[2], reduced[3], reduced[4]
}

func reduceFiveArrays(values [5][256]float32) [5]float32 {
	for stride := 128; stride > 0; stride >>= 1 {
		for threadIndex := 0; threadIndex < stride; threadIndex++ {
			for valueIndex := range 5 {
				values[valueIndex][threadIndex] += values[valueIndex][threadIndex+stride]
			}
		}
	}

	return [5]float32{values[0][0], values[1][0], values[2][0], values[3][0], values[4][0]}
}

func dagExpected(conditionals []float32) float32 {
	partialCount := metalCausalPartialCount(len(conditionals))
	partials := make([]float32, partialCount)

	for groupIndex := range partialCount {
		partials[groupIndex] = dagPartialForGroup(conditionals, groupIndex)
	}

	return dagFinalizePartials(partials)
}

func dagPartialForGroup(conditionals []float32, groupIndex int) float32 {
	values := [256]float32{}

	for threadIndex := range 256 {
		valueIndex := groupIndex*256 + threadIndex
		if valueIndex < len(conditionals) {
			values[threadIndex] = max(conditionals[valueIndex], 1.0e-12)
			continue
		}

		values[threadIndex] = 1
	}

	return reduceProductArray(values)
}

func dagFinalizePartials(partials []float32) float32 {
	values := [256]float32{}

	for threadIndex := range 256 {
		values[threadIndex] = 1
		for partialIndex := threadIndex; partialIndex < len(partials); partialIndex += 256 {
			values[threadIndex] *= partials[partialIndex]
		}
	}

	return reduceProductArray(values)
}

func reduceProductArray(values [256]float32) float32 {
	for stride := 128; stride > 0; stride >>= 1 {
		for threadIndex := 0; threadIndex < stride; threadIndex++ {
			values[threadIndex] *= values[threadIndex+stride]
		}
	}

	return values[0]
}
