package attention

import (
	"math"
)

// softmax computes in-place softmax.
func softmax(scores []float64) {
	if len(scores) == 0 {
		return
	}

	maxValue := reduceMax(scores)
	sum := 0.0

	for index, score := range scores {
		value := math.Exp(score - maxValue)
		scores[index] = value
		sum += value
	}

	divScalar(scores, sum)
}

// sdpaHead — SDPA per head. Three SIMD kernels back the per-row pipeline:
// attentionRowScores → softmaxRow → attentionRowOutput.
func sdpaHead(out, q, k, v []float64, queryLength, keyValueLength, headDim int) {
	scale := 1.0 / math.Sqrt(float64(headDim))
	scores := make([]float64, keyValueLength)

	for queryIndex := 0; queryIndex < queryLength; queryIndex++ {
		qRow := q[queryIndex*headDim : (queryIndex+1)*headDim]
		outRow := out[queryIndex*headDim : (queryIndex+1)*headDim]
		attentionRowScoresKernel(scores, qRow, k, keyValueLength, headDim, scale)

		softmax(scores)
		attentionRowOutputKernel(outRow, scores, v, keyValueLength, headDim)
	}
}

func sdpaHeadCausal(out, q, k, v []float64, queryLength, keyValueLength, headDim int) {
	scale := 1.0 / math.Sqrt(float64(headDim))
	scores := make([]float64, keyValueLength)
	offset := keyValueLength - queryLength

	for queryIndex := 0; queryIndex < queryLength; queryIndex++ {
		visible := offset + queryIndex + 1
		visibleScores := scores[:visible]
		qRow := q[queryIndex*headDim : (queryIndex+1)*headDim]
		outRow := out[queryIndex*headDim : (queryIndex+1)*headDim]
		attentionRowScoresKernel(
			visibleScores,
			qRow,
			k[:visible*headDim],
			visible,
			headDim,
			scale,
		)
		softmax(visibleScores)
		attentionRowOutputKernel(outRow, visibleScores, v, visible, headDim)
	}
}
