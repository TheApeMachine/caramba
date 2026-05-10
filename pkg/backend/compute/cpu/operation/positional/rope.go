package positional

import "math"

/*
RoPE applies Rotary Position Embeddings to Q or K tensors.

For each position t and each pair of dimensions (2i, 2i+1):

	theta_i = 1 / (base^(2i/head_dim))
	angle   = t * theta_i
	[x_{2i}, x_{2i+1}] -> [x_{2i}*cos(angle) - x_{2i+1}*sin(angle),
	                        x_{2i}*sin(angle) + x_{2i+1}*cos(angle)]

Forward: shape=[batch, num_heads, seq_len, head_dim], data[0]=input tensor.
*/
type RoPE struct {
	Base    float64 // default 10000.0
	HeadDim int
}

// NewRoPE returns a RoPE with the given base frequency and head dimension.
func NewRoPE(base float64, headDim int) *RoPE {
	if base == 0 {
		base = 10000.0
	}
	return &RoPE{Base: base, HeadDim: headDim}
}

// buildTables precomputes interleaved cos/sin tables for all (position, pair)
// combinations, ready to be consumed by the SIMD kernels.
// Layout: for each position t (0..seqLen-1) and each pair i (0..numPairs-1):
//
//	cos_table[t*numPairs + i] = cos(t * theta_i)
//	sin_table[t*numPairs + i] = sin(t * theta_i)
func (r *RoPE) buildTables(seqLen int) (cosTable, sinTable []float64) {
	numPairs := r.HeadDim / 2
	n := seqLen * numPairs
	cosTable = make([]float64, n)
	sinTable = make([]float64, n)
	for t := 0; t < seqLen; t++ {
		for i := 0; i < numPairs; i++ {
			theta := 1.0 / math.Pow(r.Base, float64(2*i)/float64(r.HeadDim))
			angle := float64(t) * theta
			cosTable[t*numPairs+i] = math.Cos(angle)
			sinTable[t*numPairs+i] = math.Sin(angle)
		}
	}
	return
}

// Forward applies RoPE to data[0].
// shape must be [batch, num_heads, seq_len, head_dim].
func (r *RoPE) Forward(shape []int, data ...[]float64) []float64 {
	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]
	if r.HeadDim == 0 {
		r.HeadDim = headDim
	}

	x := data[0]
	out := make([]float64, len(x))

	cosTable, sinTable := r.buildTables(seqLen)
	numPairs := headDim / 2

	applyRoPE(out, x, cosTable, sinTable, batch, numHeads, seqLen, numPairs)
	return out
}
