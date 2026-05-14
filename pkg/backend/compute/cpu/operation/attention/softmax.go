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
func sdpaHead(out, q, k, v []float64, seqLen, headDim int, maskFn func(i, j int) bool) {
	scale := 1.0 / math.Sqrt(float64(headDim))
	scores := make([]float64, seqLen)

	if maskFn == nil {
		for i := 0; i < seqLen; i++ {
			qRow := q[i*headDim : (i+1)*headDim]
			outRow := out[i*headDim : (i+1)*headDim]
			attentionRowScoresKernel(scores, qRow, k, seqLen, headDim, scale)
			softmax(scores)
			attentionRowOutputKernel(outRow, scores, v, seqLen, headDim)
		}

		return
	}

	for i := 0; i < seqLen; i++ {
		qRow := q[i*headDim : (i+1)*headDim]

		for j := 0; j < seqLen; j++ {
			if maskFn(i, j) {
				scores[j] = math.Inf(-1)
				continue
			}

			kRow := k[j*headDim : (j+1)*headDim]
			scores[j] = dotProduct(qRow, kRow) * scale
		}

		softmax(scores)
		outRow := out[i*headDim : (i+1)*headDim]
		attentionRowOutputKernel(outRow, scores, v, seqLen, headDim)
	}
}
