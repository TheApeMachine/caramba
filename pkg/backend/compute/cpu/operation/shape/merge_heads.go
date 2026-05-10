package shape

// MergeHeads is the inverse of ViewAsHeads.
// [B, H, T, head_dim] -> [B, T, H*head_dim]
//
// Algorithm:
//  1. Transpose dims 1 and 2: [B, H, T, head_dim] -> [B, T, H, head_dim].
//  2. Reshape (free): [B, T, H, head_dim] -> [B, T, H*head_dim].
//
// Forward(shape=[B,H,T,head_dim], data[0]) -> flat [B,T,H*head_dim] buffer.
type MergeHeads struct{}

// NewMergeHeads creates the operation.
func NewMergeHeads() *MergeHeads {
	return &MergeHeads{}
}

// Forward returns [B, T, H*head_dim] as a flat buffer.
// shape must be [B, H, T, head_dim].
func (m *MergeHeads) Forward(shape []int, data ...[]float64) []float64 {
	// Transpose dims 1 (H) and 2 (T) -> [B, T, H, head_dim].
	t := &Transpose{Dim0: 1, Dim1: 2}
	transposed := t.Forward(shape, data[0])
	// The data is now laid out as [B, T, H, head_dim] — no further copy needed
	// for the logical reshape to [B, T, H*head_dim].
	return transposed
}
