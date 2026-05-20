package metal

import (
	"math"

	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

type lossPairFixture struct {
	predictionBytes []byte
	targetBytes     []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

type lossCrossEntropyFixture struct {
	logitBytes      []byte
	targetBytes     []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

func lossPairFixtureForTest(
	name string,
	elementCount int,
	storageDType dtype.DType,
) lossPairFixture {
	predictions, targets := lossPairValues(name, elementCount)
	predictionBytes := encodeLossValuesAsDType(predictions, storageDType)
	targetBytes := encodeLossValuesAsDType(targets, storageDType)
	storedPredictions := decodeDTypeBytesToFloat32(predictionBytes, storageDType)
	storedTargets := decodeDTypeBytesToFloat32(targetBytes, storageDType)
	expected := pairLossExpectedFloat32(name, storedPredictions, storedTargets)

	return lossPairFixture{
		predictionBytes: predictionBytes,
		targetBytes:     targetBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func lossPairValues(name string, elementCount int) ([]float32, []float32) {
	if name == "binary_cross_entropy" || name == "kl_divergence" {
		return lossProbabilityValues(elementCount)
	}

	predictions := make([]float32, elementCount)
	targets := make([]float32, elementCount)

	for index := range predictions {
		predictions[index] = centeredPowerOfTwoValue(index*7+3, 41, 16)
		targets[index] = centeredPowerOfTwoValue(index*11+5, 37, 17)
	}

	return predictions, targets
}

func lossProbabilityValues(elementCount int) ([]float32, []float32) {
	predictions := make([]float32, elementCount)
	targets := make([]float32, elementCount)

	for index := range predictions {
		predictions[index] = 0.05 + 0.90*float32((index*37+11)%97)/96
		targets[index] = 0.04 + 0.92*float32((index*29+7)%89)/88
	}

	return predictions, targets
}

func pairLossExpectedFloat32(
	name string,
	predictions []float32,
	targets []float32,
) []float32 {
	scratch := pairLossPartialSums(name, predictions, targets)
	return []float32{lossFinalizeFloat32(scratch, len(predictions))}
}

func pairLossPartialSums(
	name string,
	predictions []float32,
	targets []float32,
) []float32 {
	partialCount := metalLossPartialCount(len(predictions))
	scratch := make([]float32, partialCount)

	for groupIndex := range partialCount {
		reduction := make([]float32, metalLossThreadCountGo)
		for threadIndex := range metalLossThreadCountGo {
			valueIndex := groupIndex*metalLossThreadCountGo + threadIndex
			if valueIndex < len(predictions) {
				reduction[threadIndex] = pairLossContribution(name, predictions[valueIndex], targets[valueIndex])
			}
		}

		scratch[groupIndex] = lossReduceFloat32(reduction)
	}

	return scratch
}

func pairLossContribution(name string, prediction float32, target float32) float32 {
	switch name {
	case "mse_loss":
		delta := prediction - target
		return delta * delta
	case "mae_loss":
		return float32(math.Abs(float64(prediction - target)))
	case "huber_loss":
		return huberLossContribution(prediction, target)
	case "binary_cross_entropy":
		return binaryCrossEntropyContribution(prediction, target)
	case "kl_divergence":
		return klDivergenceContribution(prediction, target)
	default:
		panic("unknown pair loss: " + name)
	}
}

func huberLossContribution(prediction float32, target float32) float32 {
	delta := prediction - target
	magnitude := float32(math.Abs(float64(delta)))

	if magnitude <= 1 {
		return 0.5 * delta * delta
	}

	return magnitude - 0.5
}

func binaryCrossEntropyContribution(prediction float32, target float32) float32 {
	safePrediction := min(max(prediction, 1.0e-7), 1.0-1.0e-7)
	return -target*float32(math.Log(float64(safePrediction))) -
		(1-target)*float32(math.Log(float64(1-safePrediction)))
}

func klDivergenceContribution(prediction float32, target float32) float32 {
	safePrediction := max(prediction, 1.0e-12)
	safeTarget := max(target, 1.0e-12)
	return safeTarget * float32(math.Log(float64(safeTarget/safePrediction)))
}

func lossCrossEntropyFixtureForTest(
	classes int,
	storageDType dtype.DType,
) lossCrossEntropyFixture {
	batch := lossCrossEntropyBatch(classes)
	logits := lossCrossEntropyLogits(batch, classes)
	targets := lossCrossEntropyTargets(batch, classes)
	logitBytes := encodeLossValuesAsDType(logits, storageDType)
	storedLogits := decodeDTypeBytesToFloat32(logitBytes, storageDType)
	expected := crossEntropyExpectedFloat32(storedLogits, targets, batch, classes)

	return lossCrossEntropyFixture{
		logitBytes:      logitBytes,
		targetBytes:     int32ValuesToBytes(targets),
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func lossCrossEntropyBatch(classes int) int {
	if classes == 1 {
		return 1
	}

	return 5
}

func lossCrossEntropyLogits(batch int, classes int) []float32 {
	logits := make([]float32, batch*classes)

	for index := range logits {
		logits[index] = centeredPowerOfTwoValue(index*13+17, 53, 11)
	}

	return logits
}

func lossCrossEntropyTargets(batch int, classes int) []int32 {
	targets := make([]int32, batch)

	for index := range targets {
		targets[index] = int32((index*3 + 1) % classes)
	}

	return targets
}

func crossEntropyExpectedFloat32(
	logits []float32,
	targets []int32,
	batch int,
	classes int,
) []float32 {
	scratch := make([]float32, batch)

	for rowIndex := range batch {
		scratch[rowIndex] = crossEntropyRowExpectedFloat32(logits, targets, rowIndex, classes)
	}

	return []float32{lossFinalizeFloat32(scratch, batch)}
}

func crossEntropyRowExpectedFloat32(
	logits []float32,
	targets []int32,
	rowIndex int,
	classes int,
) float32 {
	rowOffset := rowIndex * classes
	reduction := crossEntropyMaxReduction(logits, rowOffset, classes)
	maximum := lossReduceMaxFloat32(reduction)
	reduction = crossEntropySumReduction(logits, rowOffset, classes, maximum)
	sum := lossReduceFloat32(reduction)
	targetLogit := logits[rowOffset+int(targets[rowIndex])]

	return -(targetLogit - maximum - float32(math.Log(float64(sum))))
}

func crossEntropyMaxReduction(logits []float32, rowOffset int, classes int) []float32 {
	reduction := make([]float32, metalLossThreadCountGo)

	for threadIndex := range metalLossThreadCountGo {
		reduction[threadIndex] = -math.MaxFloat32
		for col := threadIndex; col < classes; col += metalLossThreadCountGo {
			reduction[threadIndex] = max(reduction[threadIndex], logits[rowOffset+col])
		}
	}

	return reduction
}

func crossEntropySumReduction(
	logits []float32,
	rowOffset int,
	classes int,
	maximum float32,
) []float32 {
	reduction := make([]float32, metalLossThreadCountGo)

	for threadIndex := range metalLossThreadCountGo {
		for col := threadIndex; col < classes; col += metalLossThreadCountGo {
			shifted := logits[rowOffset+col] - maximum
			reduction[threadIndex] += float32(math.Exp(float64(shifted)))
		}
	}

	return reduction
}

func lossFinalizeFloat32(scratch []float32, denominator int) float32 {
	reduction := make([]float32, metalLossThreadCountGo)

	for threadIndex := range metalLossThreadCountGo {
		for index := threadIndex; index < len(scratch); index += metalLossThreadCountGo {
			reduction[threadIndex] += scratch[index]
		}
	}

	return lossReduceFloat32(reduction) / float32(denominator)
}

func lossReduceFloat32(reduction []float32) float32 {
	for stride := metalLossThreadCountGo / 2; stride > 0; stride >>= 1 {
		for index := range stride {
			reduction[index] += reduction[index+stride]
		}
	}

	return reduction[0]
}

func lossReduceMaxFloat32(reduction []float32) float32 {
	for stride := metalLossThreadCountGo / 2; stride > 0; stride >>= 1 {
		for index := range stride {
			reduction[index] = max(reduction[index], reduction[index+stride])
		}
	}

	return reduction[0]
}

func encodeLossValuesAsDType(values []float32, storageDType dtype.DType) []byte {
	if storageDType == dtype.Float32 {
		return dtypeconvert.Float32ToBytes(values)
	}

	return encodeFloat32ValuesAsDType(values, storageDType)
}
