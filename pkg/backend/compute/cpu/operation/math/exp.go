package math

import gomath "math"

// Exp computes exp(x) elementwise. Pure Go — no SIMD for float64 exp.
type Exp struct{}

func NewExp() *Exp { return &Exp{} }

func (op *Exp) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = gomath.Exp(v)
	}
	return out
}
