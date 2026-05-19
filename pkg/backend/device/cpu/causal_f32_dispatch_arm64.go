//go:build arm64

package cpu

func CateFloat32Native(treated, control, out []float32) {
	SubFloat32Native(out, treated, control)
}

func CounterfactualFloat32Native(
	out, observedY, observedX, counterfactualX []float32,
	slope float32,
) {
	elementCount := len(out)
	blockCount := elementCount &^ 3

	if blockCount > 0 {
		CounterfactualF32NEONAsm(
			&out[0], &observedY[0], &observedX[0], &counterfactualX[0],
			slope, blockCount,
		)
	}

	for index := blockCount; index < elementCount; index++ {
		out[index] = observedY[index] + slope*(counterfactualX[index]-observedX[index])
	}
}

func DoInterveneFloat32Native(out, adjacency []float32, intervened []int32, nodeCount int) {
	copy(out, adjacency)

	for _, nodeID := range intervened {
		target := int(nodeID)

		if target < 0 || target >= nodeCount {
			continue
		}

		for sourceIndex := 0; sourceIndex < nodeCount; sourceIndex++ {
			out[sourceIndex*nodeCount+target] = 0
		}
	}
}

func BackdoorAdjustmentFloat32Native(
	conditional, marginalZ, out []float32,
	xCount, zCount, yCount int,
) {
	for index := range out {
		out[index] = 0
	}

	if yCount == 1 {
		for xIndex := 0; xIndex < xCount; xIndex++ {
			rowOffset := xIndex * zCount
			out[xIndex] = DotFloat32Native(conditional[rowOffset:rowOffset+zCount], marginalZ)
		}

		return
	}

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			baseIndex := xIndex*zCount*yCount + yIndex
			out[xIndex*yCount+yIndex] = StridedDotF32NEONAsm(
				&conditional[baseIndex], yCount, &marginalZ[0], zCount,
			)
		}
	}
}

func IvEstimateFloat32Native(instrument, treatment, outcome []float32) float32 {
	elementCount := len(instrument)
	meanZ := SumFloat32Native(instrument) / float32(elementCount)
	meanX := SumFloat32Native(treatment) / float32(elementCount)
	meanY := SumFloat32Native(outcome) / float32(elementCount)

	covZY := DotFloat32Native(instrument, outcome) - meanZ*meanY*float32(elementCount)
	covZX := DotFloat32Native(instrument, treatment) - meanZ*meanX*float32(elementCount)

	if covZX == 0 {
		return 0
	}

	return covZY / covZX
}

func MarkovFlowFloat32Native(
	mi []float32,
	partition []int32,
	out []float32,
	nodeCount int,
	targetLabel int32,
) {
	for nodeIndex := 0; nodeIndex < nodeCount; nodeIndex++ {
		if partition[nodeIndex] != targetLabel {
			out[nodeIndex] = 0
			continue
		}

		var sum float32

		for otherIndex := 0; otherIndex < nodeCount; otherIndex++ {
			if partition[otherIndex] != 0 {
				continue
			}

			sum += mi[nodeIndex*nodeCount+otherIndex]
		}

		out[nodeIndex] = sum
	}
}

func FrontdoorAdjustmentFloat32Native(
	mediatorGivenX, outcomeGivenXM, marginalX, out []float32,
	xCount, mCount, yCount int,
) {
	for index := range out {
		out[index] = 0
	}

	stride := mCount * yCount

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			var total float32

			for mIndex := 0; mIndex < mCount; mIndex++ {
				pmx := mediatorGivenX[xIndex*mCount+mIndex]
				baseIndex := mIndex*yCount + yIndex
				innerSum := StridedDotF32NEONAsm(
					&outcomeGivenXM[baseIndex], stride, &marginalX[0], xCount,
				)
				total += pmx * innerSum
			}

			out[xIndex*yCount+yIndex] = total
		}
	}
}
