package shape

// ViewAsHeads reshapes [B, T, D] -> [B, H, T, D/H] in flat row-major layout.
//
// Algorithm:
//  1. Interpret input as [B, T, H, head_dim]  (reshape — no data move).
//  2. Transpose dims 1 and 2 -> [B, H, T, head_dim].
//
// Forward(shape=[B,T,D], data[0]) -> flat [B,H,T,D/H] buffer.
type ViewAsHeads struct {
	NumHeads int
}

// NewViewAsHeads creates the operation.
func NewViewAsHeads(numHeads int) *ViewAsHeads {
	return &ViewAsHeads{NumHeads: numHeads}
}

// Forward returns [B,H,T,head_dim] as a flat buffer.
// shape must be [B, T, D] where D is divisible by NumHeads.
func (v *ViewAsHeads) Forward(shape []int, data ...[]float64) []float64 {
	B, T, D := shape[0], shape[1], shape[2]
	H := v.NumHeads
	headDim := D / H

	// Step 1: treat the flat buffer as [B, T, H, head_dim] (free reshape).
	// Step 2: transpose dims 1 and 2 to get [B, H, T, head_dim].
	t := &Transpose{Dim0: 1, Dim1: 2}
	return t.Forward([]int{B, T, H, headDim}, data[0])
}
