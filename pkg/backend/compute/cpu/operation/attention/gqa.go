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

	batch, numHeads, numKVHeads, seqLen, headDim, err := stateDict.GQALayout("attention.gqa")

	if err != nil {
		return nil, err
	}

	Q, K, V := stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2]
	keyValueWidth := batch * numKVHeads * headDim

	if len(K) != len(V) || keyValueWidth == 0 || len(K)%keyValueWidth != 0 {
		return nil, fmt.Errorf("attention.gqa: K/V lengths must match whole cached heads")
	}

	keyValueLen := len(K) / keyValueWidth
	keyShape := []int{batch, numKVHeads, seqLen, headDim}

	if stateDict.KVCache != nil {
		if stateDict.NodeID == "" {
			return nil, fmt.Errorf("attention.gqa: node id is required for KV cache")
		}

		var err error

		K, V, keyShape, err = stateDict.KVCache.Append(stateDict.NodeID, keyShape, K, V)

		if err != nil {
			return nil, err
		}

		keyValueLen = keyShape[2]
	}

	if keyValueLen < seqLen {
		return nil, fmt.Errorf(
			"attention.gqa: key/value length %d is shorter than query length %d",
			keyValueLen,
			seqLen,
		)
	}

	queryHeadStride := seqLen * headDim
	keyValueHeadStride := keyValueLen * headDim
	groups := numHeads / numKVHeads

	qBatchStride := numHeads * queryHeadStride
	kvBatchStride := numKVHeads * keyValueHeadStride
	total := batch * numHeads * queryHeadStride

	if len(Q) != total || len(K) != batch*kvBatchStride || len(V) != batch*kvBatchStride {
		return nil, fmt.Errorf("attention.gqa: Q/K/V lengths do not match GQA shape")
	}

	stateDict.EnsureOperationOutLen(total)

	for b := 0; b < batch; b++ {
		for kv := 0; kv < numKVHeads; kv++ {
			kvOff := b*kvBatchStride + kv*keyValueHeadStride

			for g := 0; g < groups; g++ {
				h := kv*groups + g
				qOff := b*qBatchStride + h*queryHeadStride
				oOff := qOff

				if stateDict.Causal {
					sdpaHeadCausal(
						stateDict.Out[oOff:oOff+queryHeadStride],
						Q[qOff:qOff+queryHeadStride],
						K[kvOff:kvOff+keyValueHeadStride],
						V[kvOff:kvOff+keyValueHeadStride],
						seqLen,
						keyValueLen,
						headDim,
					)

					continue
				}

				sdpaHead(
					stateDict.Out[oOff:oOff+queryHeadStride],
					Q[qOff:qOff+queryHeadStride],
					K[kvOff:kvOff+keyValueHeadStride],
					V[kvOff:kvOff+keyValueHeadStride],
					seqLen,
					keyValueLen,
					headDim,
				)
			}
		}
	}

	return stateDict, nil
}
