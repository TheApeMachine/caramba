//go:build !arm64

package kernels

import "math"

func cateFloat32Native(treated, control, out []float32) {
	for index := range out {
		out[index] = treated[index] - control[index]
	}
}

func counterfactualFloat32Native(
	out, observedY, observedX, counterfactualX []float32,
	slope float32,
) {
	for index := range out {
		out[index] = observedY[index] + slope*(counterfactualX[index]-observedX[index])
	}
}

func doInterveneFloat32Native(out, adjacency []float32, intervened []int32, nodeCount int) {
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

func backdoorAdjustmentFloat32Native(
	conditional, marginalZ, out []float32,
	xCount, zCount, yCount int,
) {
	for index := range out {
		out[index] = 0
	}

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			var sum float32

			for zIndex := 0; zIndex < zCount; zIndex++ {
				condIndex := (xIndex*zCount+zIndex)*yCount + yIndex
				sum += conditional[condIndex] * marginalZ[zIndex]
			}

			out[xIndex*yCount+yIndex] = sum
		}
	}
}

func frontdoorAdjustmentFloat32Native(
	mediatorGivenX, outcomeGivenXM, marginalX, out []float32,
	xCount, mCount, yCount int,
) {
	for index := range out {
		out[index] = 0
	}

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			var total float32

			for mIndex := 0; mIndex < mCount; mIndex++ {
				pmx := mediatorGivenX[xIndex*mCount+mIndex]

				var innerSum float32

				for xPrimeIndex := 0; xPrimeIndex < xCount; xPrimeIndex++ {
					outcomeIndex := (xPrimeIndex*mCount+mIndex)*yCount + yIndex
					innerSum += outcomeGivenXM[outcomeIndex] * marginalX[xPrimeIndex]
				}

				total += pmx * innerSum
			}

			out[xIndex*yCount+yIndex] = total
		}
	}
}

func ivEstimateFloat32Native(instrument, treatment, outcome []float32) float32 {
	elementCount := len(instrument)
	var meanZ, meanX, meanY float64

	for index := 0; index < elementCount; index++ {
		meanZ += float64(instrument[index])
		meanX += float64(treatment[index])
		meanY += float64(outcome[index])
	}

	meanZ /= float64(elementCount)
	meanX /= float64(elementCount)
	meanY /= float64(elementCount)

	var covZY, covZX float64

	for index := 0; index < elementCount; index++ {
		deltaZ := float64(instrument[index]) - meanZ
		deltaY := float64(outcome[index]) - meanY
		deltaX := float64(treatment[index]) - meanX
		covZY += deltaZ * deltaY
		covZX += deltaZ * deltaX
	}

	if math.Abs(covZX) < 1e-12 {
		return 0
	}

	return float32(covZY / covZX)
}

func markovFlowFloat32Native(
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
