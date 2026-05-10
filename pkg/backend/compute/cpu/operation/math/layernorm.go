package math

import gomath "math"

// LayerNorm normalizes over the last dimension.
// data[0]=x [batch*seq*d_model], shape=[batch, seq, d_model].
// y = (x - mean) / sqrt(var + eps) * Weight + Bias.
type LayerNorm struct {
	Eps    float64
	Weight []float64 // gamma, length d_model
	Bias   []float64 // beta, length d_model
}

func NewLayerNorm(eps float64, weight, bias []float64) *LayerNorm {
	return &LayerNorm{Eps: eps, Weight: weight, Bias: bias}
}

func (op *LayerNorm) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	dModel := shape[len(shape)-1]
	n := len(x) / dModel
	out := make([]float64, len(x))

	for i := 0; i < n; i++ {
		row := x[i*dModel : (i+1)*dModel]
		o := out[i*dModel : (i+1)*dModel]

		// mean via SIMD sum
		mean := reduceSum(row) / float64(dModel)

		// variance: sum((x-mean)^2)/d
		// compute shifted row then sum of squares
		tmp := make([]float64, dModel)
		for j, v := range row {
			tmp[j] = v - mean
		}
		variance := reduceSumSq(tmp) / float64(dModel)
		invStd := 1.0 / gomath.Sqrt(variance+op.Eps)

		// normalize and affine
		for j := 0; j < dModel; j++ {
			o[j] = tmp[j]*invStd*op.Weight[j] + op.Bias[j]
		}
	}
	return out
}
