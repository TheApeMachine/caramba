//go:build arm64

package attention

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/causal"
)

func ComputeHeadScoresNative(
	queryView, keyView []float32,
	qIndex, seqK, headDim int,
	queryHeadOffset, kvHeadOffset int,
	queryStride, kvStride int,
	scale float32,
	scores []float32,
	config MultiHeadAttentionConfig,
) {
	queryHead := queryView[qIndex*queryStride+queryHeadOffset : qIndex*queryStride+queryHeadOffset+headDim]

	for kIndex := range seqK {
		keyHead := keyView[kIndex*kvStride+kvHeadOffset : kIndex*kvStride+kvHeadOffset+headDim]
		score := DotFloat32Native(queryHead, keyHead) * scale

		if config.Causal && kIndex > qIndex {
			score = float32(math.Inf(-1))
		}

		if config.WindowSize > 0 && qIndex-kIndex >= config.WindowSize {
			score = float32(math.Inf(-1))
		}

		if config.ALiBiSlope != 0 {
			score += config.ALiBiSlope * float32(kIndex-qIndex)
		}

		scores[kIndex] = score
	}
}

func StableSoftmaxRowNative(scores []float32) {
	if len(scores) == 0 {
		return
	}

	maximum := ReduceMaxFloat32Native(scores)
	sum := SoftmaxRowFillExpsNative(scores, scores, maximum)
	normalizeRow(scores, sum)
}

func WriteHeadOutputNative(
	scores, valueView, outView []float32,
	qIndex, seqK, headDim int,
	queryHeadOffset, kvHeadOffset int,
	queryStride, kvStride int,
) {
	outBase := qIndex*queryStride + queryHeadOffset

	for dimIndex := range headDim {
		outView[outBase+dimIndex] = causal.StridedDotF32NEONAsm(
			&valueView[kvHeadOffset+dimIndex],
			kvStride,
			&scores[0],
			seqK,
		)
	}
}

func ApplyAttentionSoftmaxNative(scores []float32, seqQ, seqK int) {
	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		row := scores[rowIndex*seqK : (rowIndex+1)*seqK]
		maximum := ReduceMaxFloat32Native(row)
		sum := SoftmaxRowFillExpsNative(row, row, maximum)
		normalizeRow(row, sum)
	}
}

func normalizeRow(row []float32, sum float32) {
	if sum == 0 {
		return
	}

	for index := range row {
		row[index] /= sum
	}
}
