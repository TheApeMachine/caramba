package masking

import "math"

// CausalMask generates a causal (lower-triangular) attention mask.
//
// Forward accepts:
//   - shape = [seq_len] or [batch, seq_len, seq_len]  (seq_len is shape[len(shape)-1])
//   - data: unused (pass nil or empty)
//
// Returns a flat [seq_len*seq_len] slice where:
//
//	value[i*seq_len+j] = 0.0   if j <= i   (attend)
//	value[i*seq_len+j] = -inf  if j >  i   (masked)
type CausalMask struct{}

func NewCausalMask() *CausalMask { return &CausalMask{} }

func (op *CausalMask) Forward(shape []int, data ...[]float64) []float64 {
	seqLen := shape[len(shape)-1]
	out := make([]float64, seqLen*seqLen)
	applyCausalMask(out, seqLen)
	return out
}

// causalMaskScalar is the pure-Go fallback used when no SIMD path covers
// the tail elements or when neither AVX2 nor NEON is available.
func causalMaskScalar(dst []float64, seqLen int) {
	ninf := math.Inf(-1)
	for i := 0; i < seqLen; i++ {
		base := i * seqLen
		for j := 0; j <= i; j++ {
			dst[base+j] = 0.0
		}
		for j := i + 1; j < seqLen; j++ {
			dst[base+j] = ninf
		}
	}
}
