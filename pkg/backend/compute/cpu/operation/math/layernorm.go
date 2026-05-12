package math

// LayerNorm normalizes over the last dimension.
// data[0]=x [batch*seq*d_model], shape=[batch, seq, d_model].
// y = (x - mean) / sqrt(var + eps) * Weight + Bias.
// Each row runs through a fused AVX2/SSE2/NEON kernel — mean, variance,
// sqrt, normalize, and affine transform all in one assembly pass.
type LayerNorm struct {
	Eps    float64
	Weight []float64
	Bias   []float64
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
		layerNormRow(o, row, op.Weight, op.Bias, op.Eps)
	}

	return out
}
