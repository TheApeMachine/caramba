package metal

import (
	"math"

	"github.com/theapemachine/caramba/pkg/dtype"
)

type reductionFixture struct {
	inputBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

func reductionFixtureForTest(
	name string,
	elementCount int,
	storageDType dtype.DType,
) reductionFixture {
	values := reductionInputValues(name, elementCount)
	inputBytes := encodeLossValuesAsDType(values, storageDType)
	storedValues := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expected := []float32{reductionExpectedFloat32(reductionOpForName(name), storedValues)}

	return reductionFixture{
		inputBytes:      inputBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func reductionInputValues(name string, elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*17+5, 67, 19)
	}

	if name == "prod" {
		for index := range values {
			values[index] = 0.9921875 + float32((index*7)%17)/1024
		}
	}

	return values
}

func reductionOpForName(name string) metalReductionOp {
	switch name {
	case "sum":
		return metalReductionSum
	case "mean":
		return metalReductionMean
	case "prod":
		return metalReductionProd
	case "reduce_min":
		return metalReductionMin
	case "reduce_max":
		return metalReductionMax
	case "argmin":
		return metalReductionArgmin
	case "argmax":
		return metalReductionArgmax
	case "l1_norm":
		return metalReductionL1Norm
	case "l2_norm":
		return metalReductionL2Norm
	case "variance":
		return metalReductionVariance
	case "stddev":
		return metalReductionStddev
	default:
		panic("unknown reduction: " + name)
	}
}

func reductionExpectedFloat32(operation metalReductionOp, values []float32) float32 {
	partialA, partialB := reductionPartialScratch(operation, values)
	return reductionFinalizeExpected(operation, partialA, partialB, len(values))
}

func reductionPartialScratch(
	operation metalReductionOp,
	values []float32,
) ([]float32, []float32) {
	partialCount := metalReductionPartialCount(len(values))
	scratchA := make([]float32, partialCount)
	scratchB := make([]float32, partialCount)

	for groupIndex := range partialCount {
		reductionA := make([]float32, metalReductionThreadCount)
		reductionB := make([]float32, metalReductionThreadCount)
		reductionFillPartial(operation, values, groupIndex, reductionA, reductionB)
		reductionReduce(operation, reductionA, reductionB)
		scratchA[groupIndex] = reductionA[0]
		scratchB[groupIndex] = reductionB[0]
	}

	return scratchA, scratchB
}

func reductionFillPartial(
	operation metalReductionOp,
	values []float32,
	groupIndex int,
	reductionA []float32,
	reductionB []float32,
) {
	for threadIndex := range metalReductionThreadCount {
		valueIndex := groupIndex*metalReductionThreadCount + threadIndex
		reductionA[threadIndex] = reductionIdentityA(operation)

		if valueIndex >= len(values) {
			continue
		}

		value := values[valueIndex]
		reductionA[threadIndex] = reductionPartialA(operation, value)
		reductionB[threadIndex] = reductionPartialB(operation, value, valueIndex)
	}
}

func reductionFinalizeExpected(
	operation metalReductionOp,
	scratchA []float32,
	scratchB []float32,
	count int,
) float32 {
	reductionA := make([]float32, metalReductionThreadCount)
	reductionB := make([]float32, metalReductionThreadCount)
	reductionFillFinalize(operation, scratchA, scratchB, reductionA, reductionB)
	reductionReduce(operation, reductionA, reductionB)
	return reductionFinalizeValue(operation, reductionA[0], reductionB[0], count)
}

func reductionFillFinalize(
	operation metalReductionOp,
	scratchA []float32,
	scratchB []float32,
	reductionA []float32,
	reductionB []float32,
) {
	for threadIndex := range metalReductionThreadCount {
		reductionA[threadIndex] = reductionFinalizeIdentityA(operation)

		for index := threadIndex; index < len(scratchA); index += metalReductionThreadCount {
			reductionCombineCandidate(
				operation,
				&reductionA[threadIndex],
				&reductionB[threadIndex],
				scratchA[index],
				scratchB[index],
			)
		}
	}
}

func reductionIdentityA(operation metalReductionOp) float32 {
	switch operation {
	case metalReductionProd:
		return 1
	case metalReductionMin, metalReductionArgmin:
		return math.MaxFloat32
	case metalReductionMax, metalReductionArgmax:
		return -math.MaxFloat32
	default:
		return 0
	}
}

func reductionFinalizeIdentityA(operation metalReductionOp) float32 {
	if reductionIsSumLike(operation) {
		return 0
	}

	return reductionIdentityA(operation)
}

func reductionPartialA(operation metalReductionOp, value float32) float32 {
	switch operation {
	case metalReductionL1Norm:
		return float32(math.Abs(float64(value)))
	case metalReductionL2Norm, metalReductionVariance, metalReductionStddev:
		return value * value
	default:
		return value
	}
}

func reductionPartialB(operation metalReductionOp, value float32, valueIndex int) float32 {
	switch operation {
	case metalReductionArgmin, metalReductionArgmax:
		return float32(valueIndex)
	case metalReductionVariance, metalReductionStddev:
		return value
	default:
		return 0
	}
}

func reductionReduce(operation metalReductionOp, reductionA []float32, reductionB []float32) {
	for stride := metalReductionThreadCount / 2; stride > 0; stride >>= 1 {
		for index := range stride {
			reductionCombineCandidate(
				operation,
				&reductionA[index],
				&reductionB[index],
				reductionA[index+stride],
				reductionB[index+stride],
			)
		}
	}
}

func reductionCombineCandidate(
	operation metalReductionOp,
	currentA *float32,
	currentB *float32,
	candidateA float32,
	candidateB float32,
) {
	switch operation {
	case metalReductionProd:
		*currentA *= candidateA
	case metalReductionMin:
		*currentA = min(*currentA, candidateA)
	case metalReductionMax:
		*currentA = max(*currentA, candidateA)
	case metalReductionArgmin:
		reductionCombineArg(currentA, currentB, candidateA, candidateB, false)
	case metalReductionArgmax:
		reductionCombineArg(currentA, currentB, candidateA, candidateB, true)
	default:
		*currentA += candidateA
		*currentB += candidateB
	}
}

func reductionCombineArg(
	currentA *float32,
	currentB *float32,
	candidateA float32,
	candidateB float32,
	useMax bool,
) {
	takeCandidate := candidateA < *currentA
	if useMax {
		takeCandidate = candidateA > *currentA
	}

	if !takeCandidate {
		return
	}

	*currentA = candidateA
	*currentB = candidateB
}

func reductionFinalizeValue(
	operation metalReductionOp,
	accumulatedA float32,
	accumulatedB float32,
	count int,
) float32 {
	switch operation {
	case metalReductionMean:
		return accumulatedA / float32(count)
	case metalReductionArgmin, metalReductionArgmax:
		return accumulatedB
	case metalReductionL2Norm:
		return float32(math.Sqrt(float64(accumulatedA)))
	case metalReductionVariance:
		return reductionVarianceValue(accumulatedA, accumulatedB, count)
	case metalReductionStddev:
		return float32(math.Sqrt(float64(reductionVarianceValue(accumulatedA, accumulatedB, count))))
	default:
		return accumulatedA
	}
}

func reductionVarianceValue(accumulatedSquare float32, accumulatedSum float32, count int) float32 {
	mean := accumulatedSum / float32(count)
	variance := accumulatedSquare/float32(count) - mean*mean

	if variance < 0 {
		return 0
	}

	return variance
}

func reductionIsSumLike(operation metalReductionOp) bool {
	switch operation {
	case metalReductionSum, metalReductionMean, metalReductionL1Norm,
		metalReductionL2Norm, metalReductionVariance, metalReductionStddev:
		return true
	default:
		return false
	}
}
