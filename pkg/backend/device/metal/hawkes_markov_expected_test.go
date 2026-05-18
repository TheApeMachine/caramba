package metal

import (
	"math"

	"github.com/theapemachine/caramba/pkg/dtype"
)

type hawkesMarkovFixture struct {
	firstBytes      []byte
	secondBytes     []byte
	thirdBytes      []byte
	fourthBytes     []byte
	fifthBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
	labels          []int32
	expectedLabels  []int32
	rows            int
	cols            int
}

func hawkesIntensityFixtureForTest(
	storageDType dtype.DType,
	eventCount int,
	queryCount int,
) hawkesMarkovFixture {
	events := hawkesEventTimes(eventCount)
	queries := hawkesQueryTimes(queryCount)
	scalars := hawkesScalars()
	eventBytes := encodeResearchValuesAsDType(events, storageDType)
	queryBytes := encodeResearchValuesAsDType(queries, storageDType)
	scalarBytes := hawkesScalarBytes(scalars, storageDType)
	storedEvents := decodeDTypeBytesToFloat32(eventBytes, storageDType)
	storedQueries := decodeDTypeBytesToFloat32(queryBytes, storageDType)
	storedScalars := hawkesStoredScalars(scalarBytes, storageDType)
	expected := hawkesIntensityExpected(storedEvents, storedQueries, storedScalars)

	return hawkesMarkovFixture{
		firstBytes: eventBytes, secondBytes: queryBytes, thirdBytes: scalarBytes[0],
		fourthBytes: scalarBytes[1], fifthBytes: scalarBytes[2],
		expectedBytes: encodeResearchValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func hawkesKernelMatrixFixtureForTest(storageDType dtype.DType, eventCount int) hawkesMarkovFixture {
	events := hawkesEventTimes(eventCount)
	scalars := hawkesScalars()
	eventBytes := encodeResearchValuesAsDType(events, storageDType)
	scalarBytes := hawkesScalarBytes(scalars, storageDType)
	storedEvents := decodeDTypeBytesToFloat32(eventBytes, storageDType)
	storedScalars := hawkesStoredScalars(scalarBytes, storageDType)
	expected := hawkesKernelMatrixExpected(storedEvents, storedScalars)

	return hawkesMarkovFixture{
		firstBytes: eventBytes, thirdBytes: scalarBytes[1], fourthBytes: scalarBytes[2],
		expectedBytes: encodeResearchValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func hawkesLogLikelihoodFixtureForTest(storageDType dtype.DType, eventCount int) hawkesMarkovFixture {
	events := hawkesEventTimes(eventCount)
	scalars := hawkesLogScalars(events[len(events)-1] + 1.0)
	eventBytes := encodeResearchValuesAsDType(events, storageDType)
	scalarBytes := hawkesScalarBytes(scalars, storageDType)
	storedEvents := decodeDTypeBytesToFloat32(eventBytes, storageDType)
	storedScalars := hawkesStoredScalars(scalarBytes, storageDType)
	expected := []float32{hawkesLogLikelihoodExpected(storedEvents, storedScalars)}

	return hawkesMarkovFixture{
		firstBytes: eventBytes, secondBytes: scalarBytes[0], thirdBytes: scalarBytes[1],
		fourthBytes: scalarBytes[2], fifthBytes: scalarBytes[3],
		expectedBytes: encodeResearchValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func markovMutualInformationFixtureForTest(storageDType dtype.DType, rows int, cols int) hawkesMarkovFixture {
	joint := markovJointValues(rows, cols)
	jointBytes := encodeResearchValuesAsDType(joint, storageDType)
	storedJoint := decodeDTypeBytesToFloat32(jointBytes, storageDType)
	expected := []float32{markovMutualInformationExpected(storedJoint, rows, cols)}

	return hawkesMarkovFixture{
		firstBytes: jointBytes, expectedBytes: encodeResearchValuesAsDType(expected, storageDType),
		expectedFloat32: expected, rows: rows, cols: cols,
	}
}

func markovPartitionFixtureForTest(storageDType dtype.DType, nodeCount int) hawkesMarkovFixture {
	adjacency := markovAdjacencyValues(nodeCount)
	labels := markovInternalNodes(nodeCount)
	adjacencyBytes := encodeResearchValuesAsDType(adjacency, storageDType)
	storedAdjacency := decodeDTypeBytesToFloat32(adjacencyBytes, storageDType)

	return hawkesMarkovFixture{
		firstBytes: adjacencyBytes, labels: labels,
		expectedLabels: markovPartitionExpected(storedAdjacency, labels, nodeCount),
	}
}

func markovFlowFixtureForTest(name string, storageDType dtype.DType, nodeCount int) hawkesMarkovFixture {
	matrix := markovMutualInformationMatrix(nodeCount)
	labels := markovPartitionLabels(nodeCount)
	matrixBytes := encodeResearchValuesAsDType(matrix, storageDType)
	storedMatrix := decodeDTypeBytesToFloat32(matrixBytes, storageDType)
	target := int32(2)
	if name == "markov_flow_internal" {
		target = 0
	}

	expected := markovFlowExpected(storedMatrix, labels, nodeCount, target)
	return hawkesMarkovFixture{
		firstBytes: matrixBytes, labels: labels,
		expectedBytes: encodeResearchValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func hawkesIntensityExpected(events []float32, queries []float32, scalars []float32) []float32 {
	out := make([]float32, len(queries))

	for queryIndex, queryTime := range queries {
		reduction := make([]float32, metalHawkesMarkovThreadCountGo)
		for threadIndex := range metalHawkesMarkovThreadCountGo {
			for eventIndex := threadIndex; eventIndex < len(events); eventIndex += metalHawkesMarkovThreadCountGo {
				if events[eventIndex] <= queryTime {
					reduction[threadIndex] += scalars[1] *
						float32(math.Exp(float64(-scalars[2]*(queryTime-events[eventIndex]))))
				}
			}
		}

		out[queryIndex] = scalars[0] + activeReduceFloat32(reduction)
	}

	return out
}

func hawkesKernelMatrixExpected(events []float32, scalars []float32) []float32 {
	out := make([]float32, len(events)*len(events))

	for rowIndex := range events {
		for colIndex := range events {
			if colIndex >= rowIndex {
				continue
			}

			delta := events[rowIndex] - events[colIndex]
			out[rowIndex*len(events)+colIndex] = scalars[1] *
				float32(math.Exp(float64(-scalars[2]*delta)))
		}
	}

	return out
}

func hawkesLogLikelihoodExpected(events []float32, scalars []float32) float32 {
	partialCount := metalHawkesMarkovPartialCount(len(events))
	scratch := make([]float32, partialCount)

	for groupIndex := range partialCount {
		reduction := make([]float32, metalHawkesMarkovThreadCountGo)
		for threadIndex := range metalHawkesMarkovThreadCountGo {
			eventIndex := groupIndex*metalHawkesMarkovThreadCountGo + threadIndex
			if eventIndex < len(events) {
				reduction[threadIndex] = hawkesLogContribution(events, scalars, eventIndex)
			}
		}

		scratch[groupIndex] = activeReduceFloat32(reduction)
	}

	return activeFinalizeScalar(scratch) - scalars[1]*scalars[0]
}

func hawkesLogContribution(events []float32, scalars []float32, eventIndex int) float32 {
	intensity := scalars[1]
	for previousIndex := range eventIndex {
		delta := events[eventIndex] - events[previousIndex]
		intensity += scalars[2] * float32(math.Exp(float64(-scalars[3]*delta)))
	}

	compensator := (scalars[2] / scalars[3]) *
		(1 - float32(math.Exp(float64(-scalars[3]*(scalars[0]-events[eventIndex])))))
	return float32(math.Log(float64(max(intensity, 1.0e-12)))) - compensator
}

func markovMutualInformationExpected(joint []float32, rows int, cols int) float32 {
	partialCount := metalHawkesMarkovPartialCount(rows * cols)
	scratch := make([]float32, partialCount)

	for groupIndex := range partialCount {
		reduction := make([]float32, metalHawkesMarkovThreadCountGo)
		for threadIndex := range metalHawkesMarkovThreadCountGo {
			flatIndex := groupIndex*metalHawkesMarkovThreadCountGo + threadIndex
			if flatIndex < rows*cols {
				reduction[threadIndex] = markovMutualInformationContribution(joint, rows, cols, flatIndex)
			}
		}

		scratch[groupIndex] = activeReduceFloat32(reduction)
	}

	return activeFinalizeScalar(scratch)
}

func markovMutualInformationContribution(joint []float32, rows int, cols int, flatIndex int) float32 {
	jointValue := joint[flatIndex]
	if jointValue <= 1.0e-12 {
		return 0
	}

	row := flatIndex / cols
	col := flatIndex - row*cols
	var marginalRow float32
	var marginalCol float32

	for colIndex := range cols {
		marginalRow += joint[row*cols+colIndex]
	}

	for rowIndex := range rows {
		marginalCol += joint[rowIndex*cols+col]
	}

	return jointValue * float32(math.Log(float64(jointValue/(marginalRow*marginalCol+1.0e-12))))
}

func markovPartitionExpected(adjacency []float32, internal []int32, nodeCount int) []int32 {
	out := make([]int32, nodeCount)

	for nodeIndex := range nodeCount {
		out[nodeIndex] = markovPartitionLabel(adjacency, internal, nodeCount, nodeIndex)
	}

	return out
}

func markovPartitionLabel(adjacency []float32, internal []int32, nodeCount int, nodeIndex int) int32 {
	if markovContainsNode(internal, nodeIndex, nodeCount) {
		return 0
	}

	incoming, outgoing := false, false
	for _, nodeID := range internal {
		if nodeID < 0 || int(nodeID) >= nodeCount {
			continue
		}

		other := int(nodeID)
		incoming = incoming || adjacency[other*nodeCount+nodeIndex] != 0
		outgoing = outgoing || adjacency[nodeIndex*nodeCount+other] != 0
	}

	if incoming && outgoing {
		return 2
	}

	if outgoing {
		return 1
	}

	return 3
}

func markovFlowExpected(matrix []float32, labels []int32, nodeCount int, targetLabel int32) []float32 {
	out := make([]float32, nodeCount)

	for nodeIndex := range nodeCount {
		if labels[nodeIndex] != targetLabel {
			continue
		}

		for otherIndex := range nodeCount {
			if labels[otherIndex] == 0 {
				out[nodeIndex] += matrix[nodeIndex*nodeCount+otherIndex]
			}
		}
	}

	return out
}

func hawkesScalarBytes(scalars []float32, storageDType dtype.DType) [][]byte {
	out := make([][]byte, len(scalars))

	for index, value := range scalars {
		out[index] = encodeResearchValuesAsDType([]float32{value}, storageDType)
	}

	return out
}

func hawkesStoredScalars(bytes [][]byte, storageDType dtype.DType) []float32 {
	out := make([]float32, len(bytes))

	for index, scalarBytes := range bytes {
		out[index] = decodeDTypeBytesToFloat32(scalarBytes, storageDType)[0]
	}

	return out
}

func hawkesEventTimes(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = float32(index+1) / 16
	}

	return values
}

func hawkesQueryTimes(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = float32(index+1)/16 + 0.03125
	}

	return values
}

func hawkesScalars() []float32 {
	return []float32{0.25, 0.125, 0.375}
}

func hawkesLogScalars(totalTime float32) []float32 {
	return []float32{totalTime, 0.25, 0.125, 0.375}
}

func hawkesMatrixEventCount(elementCount int) int {
	count := int(math.Sqrt(float64(elementCount)))
	if count < 1 {
		return 1
	}

	return count
}

func markovRowsForTest(elementCount int) int {
	if elementCount < 4 {
		return 1
	}

	return 4
}

func hawkesScalarULP(storageDType dtype.DType) uint32 {
	if storageDType == dtype.Float32 {
		return 1024
	}

	return 3
}
