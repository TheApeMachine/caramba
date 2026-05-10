package attention

import "math"

// expShift sets dst[i] = exp(src[i] - offset).
// Pure Go because float64 exp has no efficient SIMD equivalent.
func expShift(dst, src []float64, offset float64) {
	for i, v := range src {
		dst[i] = math.Exp(v - offset)
	}
}

// softmax computes in-place softmax over scores.
func softmax(scores []float64) {
	max := reduceMax(scores)
	expShift(scores, scores, max)
	sum := reduceSum(scores)
	divScalar(scores, sum)
}

// sdpaHead computes scaled dot-product attention for a single head.
// q, k, v are each [seqLen * headDim].
// out must be pre-allocated [seqLen * headDim].
// maskFn(i, j) returns true if position j is masked (score set to -inf).
func sdpaHead(out, q, k, v []float64, seqLen, headDim int, maskFn func(i, j int) bool) {
	scale := 1.0 / math.Sqrt(float64(headDim))
	scores := make([]float64, seqLen)

	for i := 0; i < seqLen; i++ {
		qRow := q[i*headDim : (i+1)*headDim]
		// compute scores[j] = dot(q[i], k[j]) * scale
		for j := 0; j < seqLen; j++ {
			if maskFn != nil && maskFn(i, j) {
				scores[j] = math.Inf(-1)
			} else {
				kRow := k[j*headDim : (j+1)*headDim]
				scores[j] = dotProduct(qRow, kRow) * scale
			}
		}
		softmax(scores)
		// output[i] = sum_j weights[j] * v[j]
		outRow := out[i*headDim : (i+1)*headDim]
		for d := range outRow {
			outRow[d] = 0
		}
		for j := 0; j < seqLen; j++ {
			vRow := v[j*headDim : (j+1)*headDim]
			scaledAdd(outRow, vRow, scores[j])
		}
	}
}
