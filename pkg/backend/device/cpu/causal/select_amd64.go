//go:build amd64

package causal

import (
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/dot"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/elementwise"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/reduction"
	"golang.org/x/sys/cpu"
)

func CateFloat32Native(treated, control, out []float32) {
	if cpu.X86.HasAVX512F {
		cateF32AVX512(treated, control, out)

		return
	}

	elementwise.SubFloat32Native(out, treated, control)
}

func CounterfactualFloat32Native(
	out, observedY, observedX, counterfactualX []float32,
	slope float32,
) {
	if cpu.X86.HasAVX512F {
		counterfactualF32AVX512(out, observedY, observedX, counterfactualX, slope)

		return
	}

	counterfactualF32Generic(out, observedY, observedX, counterfactualX, slope)
}

func DoInterveneFloat32Native(out, adjacency []float32, intervened []int32, nodeCount int) {
	doInterveneF32Generic(out, adjacency, intervened, nodeCount)
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
			out[xIndex] = dot.DotFloat32Native(conditional[rowOffset:rowOffset+zCount], marginalZ)
		}

		return
	}

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			baseIndex := xIndex*zCount*yCount + yIndex
			out[xIndex*yCount+yIndex] = stridedDotF32Native(
				conditional[baseIndex:],
				yCount,
				marginalZ,
				zCount,
			)
		}
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
				innerSum := stridedDotF32Native(
					outcomeGivenXM[baseIndex:],
					stride,
					marginalX,
					xCount,
				)
				total += pmx * innerSum
			}

			out[xIndex*yCount+yIndex] = total
		}
	}
}

func IvEstimateFloat32Native(instrument, treatment, outcome []float32) float32 {
	elementCount := len(instrument)
	meanZ := reduction.SumFloat32Native(instrument) / float32(elementCount)
	meanX := reduction.SumFloat32Native(treatment) / float32(elementCount)
	meanY := reduction.SumFloat32Native(outcome) / float32(elementCount)

	covZY := dot.DotFloat32Native(instrument, outcome) - meanZ*meanY*float32(elementCount)
	covZX := dot.DotFloat32Native(instrument, treatment) - meanZ*meanX*float32(elementCount)

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
	markovFlowF32Generic(mi, partition, out, nodeCount, targetLabel)
}

func stridedDotF32Native(values []float32, stride int, weights []float32, elementCount int) float32 {
	if cpu.X86.HasAVX512F {
		return stridedDotF32AVX512(values, stride, weights, elementCount)
	}

	return stridedDotF32Generic(values, stride, weights, elementCount)
}
