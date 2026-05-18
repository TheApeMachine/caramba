package metal

import "github.com/theapemachine/caramba/pkg/dtype"

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
