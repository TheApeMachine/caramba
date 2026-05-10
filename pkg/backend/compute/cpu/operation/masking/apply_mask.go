package masking

// ApplyMask applies an additive mask to attention scores.
//
// Convention: mask[i] = 0.0 to attend, -Inf to block.
// output[i] = scores[i] + mask[i]
//
// Forward: shape=[batch, heads, seq, seq], data[0]=scores, data[1]=mask
type ApplyMask struct{}

func NewApplyMask() *ApplyMask { return &ApplyMask{} }

func (op *ApplyMask) Forward(shape []int, data ...[]float64) []float64 {
	scores := data[0]
	mask := data[1]
	out := make([]float64, len(scores))
	applyMaskAdd(out, scores, mask)
	return out
}

// applyMaskScalar is the pure-Go fallback.
func applyMaskScalar(dst, scores, mask []float64) {
	for i := range scores {
		dst[i] = scores[i] + mask[i]
	}
}
