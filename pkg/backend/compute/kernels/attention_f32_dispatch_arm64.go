//go:build arm64

package kernels

import "math"

func computeHeadScoresNative(
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
		score := dotFloat32Native(queryHead, keyHead) * scale

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

func stableSoftmaxRowNative(scores []float32) {
	if len(scores) == 0 {
		return
	}

	maximum := reduceMaxFloat32Native(scores)
	sum := softmaxRowFillExpsNative(scores, scores, maximum)
	normalizeRow(scores, sum)
}

func writeHeadOutputNative(
	scores, valueView, outView []float32,
	qIndex, seqK, headDim int,
	queryHeadOffset, kvHeadOffset int,
	queryStride, kvStride int,
) {
	outBase := qIndex*queryStride + queryHeadOffset

	for dimIndex := range headDim {
		outView[outBase+dimIndex] = stridedDotF32NEONAsm(
			&valueView[kvHeadOffset+dimIndex],
			kvStride,
			&scores[0],
			seqK,
		)
	}
}

func applyAttentionSoftmaxNative(scores []float32, seqQ, seqK int) {
	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		row := scores[rowIndex*seqK : (rowIndex+1)*seqK]
		maximum := reduceMaxFloat32Native(row)
		sum := softmaxRowFillExpsNative(row, row, maximum)
		normalizeRow(row, sum)
	}
}
