package attention

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

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

func (gqa *GQA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("attention.gqa", 3); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) != 5 {
		return nil, fmt.Errorf("attention.gqa: expected rank 5, got %d", len(shape))
	}

	batch := shape[0]
	numHeads := shape[1]
	numKVHeads := shape[2]
	seqLen := shape[3]
	headDim := shape[4]

	if batch <= 0 || numHeads <= 0 || numKVHeads <= 0 || seqLen <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("attention.gqa: all shape dimensions must be positive")
	}

	if numHeads%numKVHeads != 0 {
		return nil, fmt.Errorf("attention.gqa: num_heads must be divisible by num_kv_heads")
	}

	Q, K, V := stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2]
	headStride := seqLen * headDim
	groups := numHeads / numKVHeads

	qBatchStride := numHeads * headStride
	kvBatchStride := numKVHeads * headStride
	total := batch * numHeads * headStride

	if len(Q) != total || len(K) != batch*kvBatchStride || len(V) != batch*kvBatchStride {
		return nil, fmt.Errorf("attention.gqa: Q/K/V lengths do not match GQA shape")
	}

	stateDict.EnsureOperationOutLen(total)

	for b := 0; b < batch; b++ {
		for kv := 0; kv < numKVHeads; kv++ {
			kvOff := b*kvBatchStride + kv*headStride
			for g := 0; g < groups; g++ {
				h := kv*groups + g
				qOff := b*qBatchStride + h*headStride
				oOff := qOff
				sdpaHead(
					stateDict.Out[oOff:oOff+headStride],
					Q[qOff:qOff+headStride],
					K[kvOff:kvOff+headStride],
					V[kvOff:kvOff+headStride],
					seqLen,
					seqLen,
					headDim,
				)
			}
		}
	}

	return stateDict, nil
}
