package math

import gomath "math"

// RMSNorm computes y = x / rms(x) * Weight, where rms = sqrt(mean(x^2) + eps).
// data[0]=x, shape=[..., d_model].
type RMSNorm struct {
	Eps    float64
	Weight []float64 // gamma, length d_model
}

func NewRMSNorm(eps float64, weight []float64) *RMSNorm {
	return &RMSNorm{Eps: eps, Weight: weight}
}

func (op *RMSNorm) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	dModel := shape[len(shape)-1]
	n := len(x) / dModel
	out := make([]float64, len(x))

	for i := 0; i < n; i++ {
		row := x[i*dModel : (i+1)*dModel]
		o := out[i*dModel : (i+1)*dModel]

		sumSq := reduceSumSq(row)
		rms := gomath.Sqrt(sumSq/float64(dModel) + op.Eps)
		invRMS := 1.0 / rms

		for j := 0; j < dModel; j++ {
			o[j] = row[j] * invRMS * op.Weight[j]
		}
	}
	return out
}
