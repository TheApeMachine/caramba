package attention

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

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

func (m *MQA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("attention.mqa", 3); err != nil {
		return nil, err
	}

	batch, numHeads, seqLen, headDim, err := attentionShape4("attention.mqa", stateDict)

	if err != nil {
		return nil, err
	}

	Q, K, V := stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2]
	headStride := seqLen * headDim
	kvBatchStride := headStride
	qBatchStride := numHeads * headStride
	total := batch * numHeads * headStride

	if len(Q) != total || len(K) != batch*headStride || len(V) != batch*headStride {
		return nil, fmt.Errorf("attention.mqa: Q/K/V lengths do not match MQA shape")
	}

	stateDict.EnsureOperationOutLen(total)

	for b := 0; b < batch; b++ {
		kvOff := b * kvBatchStride
		for h := 0; h < numHeads; h++ {
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

	return stateDict, nil
}
