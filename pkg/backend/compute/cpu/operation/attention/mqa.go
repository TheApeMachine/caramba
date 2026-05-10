package attention

// MQA implements Multi-Query Attention.
// Q has num_heads heads; K and V share a single head (broadcast across Q heads).
//
// shape: [batch, num_heads, seq_len, head_dim]
// data[0]=Q [batch*num_heads*seq_len*head_dim]
// data[1]=K [batch*1*seq_len*head_dim]
// data[2]=V [batch*1*seq_len*head_dim]
// output:   [batch*num_heads*seq_len*head_dim]
type MQA struct{}

func NewMQA() *MQA { return &MQA{} }

func (m *MQA) Forward(shape []int, data ...[]float64) []float64 {
	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]

	Q, K, V := data[0], data[1], data[2]
	headStride := seqLen * headDim
	kvBatchStride := headStride
	qBatchStride := numHeads * headStride
	total := batch * numHeads * headStride
	out := make([]float64, total)

	for b := 0; b < batch; b++ {
		kvOff := b * kvBatchStride
		for h := 0; h < numHeads; h++ {
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
	return out
}
