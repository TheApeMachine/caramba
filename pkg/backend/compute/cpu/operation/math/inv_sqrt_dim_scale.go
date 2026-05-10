package math

import gomath "math"

// InvSqrtDimScale multiplies each element of data[0] by 1/sqrt(shape[-1]).
// This is the standard attention head scale factor.
type InvSqrtDimScale struct{}

func NewInvSqrtDimScale() *InvSqrtDimScale { return &InvSqrtDimScale{} }

func (op *InvSqrtDimScale) Forward(shape []int, data ...[]float64) []float64 {
	dim := shape[len(shape)-1]
	scale := 1.0 / gomath.Sqrt(float64(dim))
	out := make([]float64, len(data[0]))
	copy(out, data[0])
	mulScalar(out, scale)
	return out
}
