package math

// RMSNorm computes y = x / rms(x) * Weight where rms = sqrt(mean(x²) + eps).
// Each row is dispatched to a fused AVX2/SSE2/NEON kernel.
type RMSNorm struct {
	Eps    float64
	Weight []float64
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
		rmsNormRow(o, row, op.Weight, op.Eps)
	}

	return out
}
