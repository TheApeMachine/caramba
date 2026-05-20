package metal

import (
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

const activeThreadCountForTest = 256

type activeFixture struct {
	firstBytes      []byte
	secondBytes     []byte
	thirdBytes      []byte
	auxiliaryBytes  []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

func activeFreeEnergyFixtureForTest(storageDType dtype.DType, elementCount int) activeFixture {
	likelihood, posterior, prior := activeFreeEnergyValues(elementCount)
	auxiliaryValues := activePositiveValues(elementCount, 23, 17)
	likelihoodBytes := encodeResearchValuesAsDType(likelihood, storageDType)
	posteriorBytes := encodeResearchValuesAsDType(posterior, storageDType)
	priorBytes := encodeResearchValuesAsDType(prior, storageDType)
	auxiliaryBytes := encodeResearchValuesAsDType(auxiliaryValues, storageDType)
	storedLikelihood := decodeDTypeBytesToFloat32(likelihoodBytes, storageDType)
	storedPosterior := decodeDTypeBytesToFloat32(posteriorBytes, storageDType)
	storedPrior := decodeDTypeBytesToFloat32(priorBytes, storageDType)
	expected := []float32{activeFreeEnergyExpected(storedLikelihood, storedPosterior, storedPrior)}

	return activeFixture{
		firstBytes: likelihoodBytes, secondBytes: posteriorBytes, thirdBytes: priorBytes,
		auxiliaryBytes: auxiliaryBytes, expectedBytes: encodeResearchValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func activeExpectedFreeEnergyFixtureForTest(
	storageDType dtype.DType,
	obsCount int,
	stateCount int,
) activeFixture {
	predictedObs := activePositiveValues(obsCount, 37, 11)
	preferredObs := activePositiveValues(obsCount, 29, 7)
	predictedState := activePositiveValues(stateCount, 31, 13)
	predictedBytes := encodeResearchValuesAsDType(predictedObs, storageDType)
	preferredBytes := encodeResearchValuesAsDType(preferredObs, storageDType)
	stateBytes := encodeResearchValuesAsDType(predictedState, storageDType)
	storedPredicted := decodeDTypeBytesToFloat32(predictedBytes, storageDType)
	storedPreferred := decodeDTypeBytesToFloat32(preferredBytes, storageDType)
	storedState := decodeDTypeBytesToFloat32(stateBytes, storageDType)
	expected := []float32{activeExpectedFreeEnergyExpected(storedPredicted, storedPreferred, storedState)}

	return activeFixture{
		firstBytes: predictedBytes, secondBytes: preferredBytes, thirdBytes: stateBytes,
		expectedBytes: encodeResearchValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func activeBinaryFixtureForTest(
	name string,
	storageDType dtype.DType,
	elementCount int,
) activeFixture {
	leftValues, rightValues := activeBinaryValues(name, elementCount)
	leftBytes := encodeResearchValuesAsDType(leftValues, storageDType)
	rightBytes := encodeResearchValuesAsDType(rightValues, storageDType)
	storedLeft := decodeDTypeBytesToFloat32(leftBytes, storageDType)
	storedRight := decodeDTypeBytesToFloat32(rightBytes, storageDType)
	expected := activeBinaryExpected(name, storedLeft, storedRight)

	return activeFixture{
		firstBytes: leftBytes, secondBytes: rightBytes,
		expectedBytes: encodeResearchValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func activeFreeEnergyValues(elementCount int) ([]float32, []float32, []float32) {
	likelihood := activePositiveValues(elementCount, 37, 11)
	posterior := activePositiveValues(elementCount, 41, 5)
	prior := activePositiveValues(elementCount, 31, 19)
	return likelihood, posterior, prior
}

func activeBinaryValues(name string, elementCount int) ([]float32, []float32) {
	if name == "belief_update" {
		return activePositiveValues(elementCount, 43, 3), activePositiveValues(elementCount, 47, 17)
	}

	errors := make([]float32, elementCount)
	precision := activePositiveValues(elementCount, 29, 23)

	for index := range errors {
		errors[index] = centeredPowerOfTwoValue(index*13+7, 59, 23)
	}

	return errors, precision
}

func activePositiveValues(elementCount int, multiplier int, offset int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = 0.05 + 0.90*float32((index*multiplier+offset)%97)/96
	}

	return values
}

func activeFreeEnergyExpected(
	likelihood []float32,
	posterior []float32,
	prior []float32,
) float32 {
	partialCount := activePartialCountForTest(len(likelihood))
	scratch := make([]float32, partialCount)

	for groupIndex := range partialCount {
		reduction := make([]float32, activeThreadCountForTest)
		for threadIndex := range activeThreadCountForTest {
			valueIndex := groupIndex*activeThreadCountForTest + threadIndex
			if valueIndex < len(likelihood) {
				reduction[threadIndex] = activeFreeEnergyContribution(
					likelihood[valueIndex], posterior[valueIndex], prior[valueIndex],
				)
			}
		}

		scratch[groupIndex] = activeReduceFloat32(reduction)
	}

	return activeFinalizeScalar(scratch)
}

func activeExpectedFreeEnergyExpected(
	predictedObs []float32,
	preferredObs []float32,
	predictedState []float32,
) float32 {
	obsPartialCount := activePartialCountForTest(len(predictedObs))
	stateScratch := activeStatePartialScratch(predictedState)
	scratch := make([]float32, obsPartialCount+len(stateScratch))

	for groupIndex := range obsPartialCount {
		scratch[groupIndex] = activeObsPartialSum(predictedObs, preferredObs, groupIndex)
	}

	copy(scratch[obsPartialCount:], stateScratch)
	return activeFinalizeScalar(scratch)
}

func activeObsPartialSum(predictedObs []float32, preferredObs []float32, groupIndex int) float32 {
	reduction := make([]float32, activeThreadCountForTest)

	for threadIndex := range activeThreadCountForTest {
		valueIndex := groupIndex*activeThreadCountForTest + threadIndex
		if valueIndex < len(predictedObs) {
			reduction[threadIndex] = activePragmaticContribution(
				predictedObs[valueIndex], preferredObs[valueIndex],
			)
		}
	}

	return activeReduceFloat32(reduction)
}

func activeStatePartialScratch(predictedState []float32) []float32 {
	partialCount := activePartialCountForTest(len(predictedState))
	scratch := make([]float32, partialCount)

	for groupIndex := range partialCount {
		reduction := make([]float32, activeThreadCountForTest)
		for threadIndex := range activeThreadCountForTest {
			valueIndex := groupIndex*activeThreadCountForTest + threadIndex
			if valueIndex < len(predictedState) {
				reduction[threadIndex] = activeEpistemicContribution(predictedState[valueIndex])
			}
		}

		scratch[groupIndex] = activeReduceFloat32(reduction)
	}

	return scratch
}

func activeBinaryExpected(name string, left []float32, right []float32) []float32 {
	if name == "belief_update" {
		return activeBeliefUpdateExpected(left, right)
	}

	out := make([]float32, len(left))
	for index, value := range left {
		out[index] = value * right[index]
	}

	return out
}

func activeBeliefUpdateExpected(likelihood []float32, prior []float32) []float32 {
	partialCount := activePartialCountForTest(len(likelihood))
	scratch := make([]float32, partialCount)

	for groupIndex := range partialCount {
		reduction := make([]float32, activeThreadCountForTest)
		for threadIndex := range activeThreadCountForTest {
			valueIndex := groupIndex*activeThreadCountForTest + threadIndex
			if valueIndex < len(likelihood) {
				reduction[threadIndex] = likelihood[valueIndex] * prior[valueIndex]
			}
		}

		scratch[groupIndex] = activeReduceFloat32(reduction)
	}

	return activeBeliefNormalizeExpected(likelihood, prior, activeFinalizeScalar(scratch))
}

func activeBeliefNormalizeExpected(likelihood []float32, prior []float32, total float32) []float32 {
	out := make([]float32, len(likelihood))

	for index, value := range likelihood {
		product := value * prior[index]
		if total != 0 {
			product /= total
		}

		out[index] = product
	}

	return out
}

func activeFreeEnergyContribution(likelihood float32, posterior float32, prior float32) float32 {
	return posterior * (-activeLog(likelihood) + activeLog(posterior) - activeLog(prior))
}

func activePragmaticContribution(predicted float32, preferred float32) float32 {
	return predicted * (activeLog(predicted) - activeLog(preferred))
}

func activeEpistemicContribution(state float32) float32 {
	return -state * activeLog(state)
}

func activeLog(value float32) float32 {
	return float32(math.Log(float64(max(value, 1.0e-12))))
}

func activePartialCountForTest(elementCount int) int {
	return (elementCount + activeThreadCountForTest - 1) / activeThreadCountForTest
}

func activeFinalizeScalar(scratch []float32) float32 {
	reduction := make([]float32, activeThreadCountForTest)

	for threadIndex := range activeThreadCountForTest {
		for index := threadIndex; index < len(scratch); index += activeThreadCountForTest {
			reduction[threadIndex] += scratch[index]
		}
	}

	return activeReduceFloat32(reduction)
}

func activeReduceFloat32(reduction []float32) float32 {
	for stride := activeThreadCountForTest / 2; stride > 0; stride >>= 1 {
		for index := range stride {
			reduction[index] += reduction[index+stride]
		}
	}

	return reduction[0]
}
