package causal

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func requireCausalFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("causal: unsupported dtype")
	}
}

func Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	requireCausalFloat32(format)

	if matrixOrder == 0 {
		return
	}

	inputView := unsafe.Slice((*float32)(input), matrixOrder*matrixOrder)
	outputView := unsafe.Slice((*float32)(output), matrixOrder*matrixOrder)

	for index := range outputView {
		outputView[index] = 0
	}

	for rowIndex := 0; rowIndex < matrixOrder; rowIndex++ {
		for colIndex := 0; colIndex <= rowIndex; colIndex++ {
			sum := float64(inputView[rowIndex*matrixOrder+colIndex])

			for innerIndex := 0; innerIndex < colIndex; innerIndex++ {
				sum -= float64(outputView[rowIndex*matrixOrder+innerIndex]) *
					float64(outputView[colIndex*matrixOrder+innerIndex])
			}

			switch {
			case rowIndex == colIndex:
				if sum <= 0 {
					return
				}

				outputView[rowIndex*matrixOrder+colIndex] = float32(math.Sqrt(sum))
			default:
				outputView[rowIndex*matrixOrder+colIndex] = float32(
					sum / float64(outputView[colIndex*matrixOrder+colIndex]),
				)
			}
		}
	}
}

func BackdoorAdjustment(
	conditional, marginalZ, output unsafe.Pointer,
	xCount, zCount, yCount int,
	format dtype.DType,
) {
	requireCausalFloat32(format)

	if xCount == 0 || zCount == 0 || yCount == 0 {
		return
	}

	conditionalView := unsafe.Slice((*float32)(conditional), xCount*zCount*yCount)
	marginalView := unsafe.Slice((*float32)(marginalZ), zCount)
	outputView := unsafe.Slice((*float32)(output), xCount*yCount)

	BackdoorAdjustmentFloat32Native(
		conditionalView, marginalView, outputView,
		xCount, zCount, yCount,
	)
}

func FrontdoorAdjustment(
	mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
	xCount, mediatorCount, yCount int,
	format dtype.DType,
) {
	requireCausalFloat32(format)

	if xCount == 0 || mediatorCount == 0 || yCount == 0 {
		return
	}

	mediatorView := unsafe.Slice((*float32)(mediatorGivenX), xCount*mediatorCount)
	outcomeView := unsafe.Slice((*float32)(outcomeGivenXM), xCount*mediatorCount*yCount)
	marginalView := unsafe.Slice((*float32)(marginalX), xCount)
	outputView := unsafe.Slice((*float32)(output), xCount*yCount)

	for index := range outputView {
		outputView[index] = 0
	}

	FrontdoorAdjustmentFloat32Native(
		mediatorView, outcomeView, marginalView, outputView,
		xCount, mediatorCount, yCount,
	)
}

func DoIntervene(
	adjacency, intervened, output unsafe.Pointer,
	nodeCount, intervenedCount int,
	format dtype.DType,
) {
	requireCausalFloat32(format)

	if nodeCount == 0 {
		return
	}

	adjacencyView := unsafe.Slice((*float32)(adjacency), nodeCount*nodeCount)
	intervenedView := unsafe.Slice((*int32)(intervened), intervenedCount)
	outputView := unsafe.Slice((*float32)(output), nodeCount*nodeCount)

	DoInterveneFloat32Native(outputView, adjacencyView, intervenedView, nodeCount)
}

func CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	requireCausalFloat32(format)

	if count == 0 {
		return
	}

	treatedView := unsafe.Slice((*float32)(treated), count)
	controlView := unsafe.Slice((*float32)(control), count)
	outputView := unsafe.Slice((*float32)(output), count)

	CateFloat32Native(treatedView, controlView, outputView)
}

func Counterfactual(
	observedY, observedX, counterfactualX, output unsafe.Pointer,
	count int,
	slope float32,
	format dtype.DType,
) {
	requireCausalFloat32(format)

	if count == 0 {
		return
	}

	observedYView := unsafe.Slice((*float32)(observedY), count)
	observedXView := unsafe.Slice((*float32)(observedX), count)
	counterfactualXView := unsafe.Slice((*float32)(counterfactualX), count)
	outputView := unsafe.Slice((*float32)(output), count)

	CounterfactualFloat32Native(
		outputView, observedYView, observedXView, counterfactualXView, slope,
	)
}

func IVEstimate(
	instrument, treatment, outcome unsafe.Pointer,
	count int,
	output unsafe.Pointer,
	format dtype.DType,
) {
	requireCausalFloat32(format)

	if count < 2 {
		return
	}

	instrumentView := unsafe.Slice((*float32)(instrument), count)
	treatmentView := unsafe.Slice((*float32)(treatment), count)
	outcomeView := unsafe.Slice((*float32)(outcome), count)
	outputView := unsafe.Slice((*float32)(output), 1)

	outputView[0] = IvEstimateFloat32Native(instrumentView, treatmentView, outcomeView)
}

func DAGMarkovFactorization(
	conditionals unsafe.Pointer,
	conditionalCount int,
	output unsafe.Pointer,
	format dtype.DType,
) {
	requireCausalFloat32(format)

	if conditionalCount == 0 {
		return
	}

	conditionalView := unsafe.Slice((*float32)(conditionals), conditionalCount)
	outputView := unsafe.Slice((*float32)(output), 1)

	product := float64(1)

	for _, conditional := range conditionalView {
		product *= math.Max(1e-12, float64(conditional))
	}

	outputView[0] = float32(product)
}

func MarkovFlowActive(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	markovFlow(mutualInformation, partition, output, nodeCount, 2, format)
}

func MarkovFlowInternal(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	markovFlow(mutualInformation, partition, output, nodeCount, 0, format)
}

func markovFlow(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	targetLabel int32,
	format dtype.DType,
) {
	requireCausalFloat32(format)

	if nodeCount == 0 {
		return
	}

	miView := unsafe.Slice((*float32)(mutualInformation), nodeCount*nodeCount)
	partitionView := unsafe.Slice((*int32)(partition), nodeCount)
	outputView := unsafe.Slice((*float32)(output), nodeCount)

	MarkovFlowFloat32Native(miView, partitionView, outputView, nodeCount, targetLabel)
}
