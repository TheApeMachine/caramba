//go:build !arm64

package cpu

import "math"

func ComputeHeadScoresNative(
	queryView, keyView []float32,
	qIndex, seqK, headDim int,
	queryHeadOffset, kvHeadOffset int,
	queryStride, kvStride int,
	scale float32,
	scores []float32,
	config MultiHeadAttentionConfig,
) {
	for kIndex := range seqK {
		var dot float32

		for dimIndex := range headDim {
			dot += queryView[qIndex*queryStride+queryHeadOffset+dimIndex] *
				keyView[kIndex*kvStride+kvHeadOffset+dimIndex]
		}

		score := dot * scale

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
	maximum := scores[0]

	for _, value := range scores[1:] {
		if value > maximum {
			maximum = value
		}
	}

	var sum float32

	for index, value := range scores {
		shifted := float32(math.Exp(float64(value - maximum)))
		scores[index] = shifted
		sum += shifted
	}

	if sum == 0 {
		return
	}

	for index := range scores {
		scores[index] /= sum
	}
}

func WriteHeadOutputNative(
	scores, valueView, outView []float32,
	qIndex, seqK, headDim int,
	queryHeadOffset, kvHeadOffset int,
	queryStride, kvStride int,
) {
	for dimIndex := range headDim {
		var sum float32

		for kIndex := range seqK {
			sum += scores[kIndex] *
				valueView[kIndex*kvStride+kvHeadOffset+dimIndex]
		}

		outView[qIndex*queryStride+queryHeadOffset+dimIndex] = sum
	}
}

func ApplyAttentionSoftmaxNative(scores []float32, seqQ, seqK int) {
	applySoftmax(scores, seqQ, seqK)
}
