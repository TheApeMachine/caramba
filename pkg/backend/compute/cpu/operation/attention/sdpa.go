package attention

// SDPA implements standard scaled dot-product attention (multi-head).
//
// shape: [batch, num_heads, seq_len, head_dim]
// data[0]=Q, data[1]=K, data[2]=V  each [batch*num_heads*seq_len*head_dim]
// output: [batch*num_heads*seq_len*head_dim]
type SDPA struct{}

func NewSDPA() *SDPA { return &SDPA{} }

func (s *SDPA) Forward(shape []int, data ...[]float64) []float64 {
	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]

	Q, K, V := data[0], data[1], data[2]
	headStride := seqLen * headDim
	batchStride := numHeads * headStride
	total := batch * numHeads * headStride
	out := make([]float64, total)

	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			off := b*batchStride + h*headStride
			sdpaHead(
				out[off:off+headStride],
				Q[off:off+headStride],
				K[off:off+headStride],
				V[off:off+headStride],
				seqLen, headDim, nil,
			)
		}
	}
	return out
}
