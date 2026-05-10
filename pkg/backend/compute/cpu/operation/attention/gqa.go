package attention

// GQA implements Grouped-Query Attention.
// Q has num_heads heads; K and V have num_kv_heads heads.
// num_heads must be divisible by num_kv_heads.
// Each KV head serves (num_heads / num_kv_heads) Q heads.
//
// shape: [batch, num_heads, num_kv_heads, seq_len, head_dim]
// data[0]=Q [batch*num_heads*seq_len*head_dim]
// data[1]=K [batch*num_kv_heads*seq_len*head_dim]
// data[2]=V [batch*num_kv_heads*seq_len*head_dim]
// output:   [batch*num_heads*seq_len*head_dim]
type GQA struct{}

func NewGQA() *GQA { return &GQA{} }

func (g *GQA) Forward(shape []int, data ...[]float64) []float64 {
	batch := shape[0]
	numHeads := shape[1]
	numKVHeads := shape[2]
	seqLen := shape[3]
	headDim := shape[4]

	Q, K, V := data[0], data[1], data[2]
	headStride := seqLen * headDim
	groups := numHeads / numKVHeads

	qBatchStride := numHeads * headStride
	kvBatchStride := numKVHeads * headStride
	total := batch * numHeads * headStride
	out := make([]float64, total)

	for b := 0; b < batch; b++ {
		for kv := 0; kv < numKVHeads; kv++ {
			kvOff := b*kvBatchStride + kv*headStride
			for g := 0; g < groups; g++ {
				h := kv*groups + g
				qOff := b*qBatchStride + h*headStride
				oOff := qOff
				sdpaHead(
					out[oOff:oOff+headStride],
					Q[qOff:qOff+headStride],
					K[kvOff:kvOff+headStride],
					V[kvOff:kvOff+headStride],
					seqLen, headDim, nil,
				)
			}
		}
	}
	return out
}
