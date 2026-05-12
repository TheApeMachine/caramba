package attention

import (
	"math"

	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

// expShift sets dst[i] = exp(src[i] - offset) using vectorized exp.
func expShift(dst, src []float64, offset float64) {
	copy(dst, src)
	mathops.AddScalarVec(dst, -offset)
	mathops.ExpVec(dst, dst)
}

// softmax computes in-place softmax over scores.
func softmax(scores []float64) {
	mx := reduceMax(scores)
	expShift(scores, scores, mx)
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

		for j := 0; j < seqLen; j++ {
			if maskFn != nil && maskFn(i, j) {
				scores[j] = math.Inf(-1)
				continue
			}

			kRow := k[j*headDim : (j+1)*headDim]
			scores[j] = dotProduct(qRow, kRow) * scale
		}

		softmax(scores)
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
