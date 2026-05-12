package math

// Exp computes exp(x) elementwise via vectorized SIMD assembly (AVX2/SSE2/NEON).
type Exp struct{}

func NewExp() *Exp { return &Exp{} }

func (op *Exp) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	out := make([]float64, len(x))
	expVec(out, x)

	return out
}
