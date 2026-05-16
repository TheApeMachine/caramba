package attention

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// SDPA implements standard scaled dot-product attention (multi-head).
//
// shape: [batch, num_heads, seq_len, head_dim]
// data[0]=Q, data[1]=K, data[2]=V  each [batch*num_heads*seq_len*head_dim]
// output: [batch*num_heads*seq_len*head_dim]
type SDPA struct{}

func NewSDPA() *SDPA { return &SDPA{} }

func (s *SDPA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("attention.sdpa", 3); err != nil {
		return nil, err
	}

	batch, numHeads, queryLength, headDim, err := attentionShape4("attention.sdpa", stateDict)

	if err != nil {
		return nil, err
	}

	Q, K, V := stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2]
	queryHeadStride := queryLength * headDim
	queryBatchStride := numHeads * queryHeadStride
	queryTotal := batch * numHeads * queryHeadStride

	if len(Q) != queryTotal {
		return nil, fmt.Errorf("attention.sdpa: Q length must equal %d", queryTotal)
	}

	keyValueWidth := batch * numHeads * headDim

	if len(K) != len(V) || keyValueWidth == 0 || len(K)%keyValueWidth != 0 {
		return nil, fmt.Errorf("attention.sdpa: K/V lengths must match whole cached heads")
	}

	keyValueLength := len(K) / keyValueWidth

	if keyValueLength < queryLength {
		return nil, fmt.Errorf(
			"attention.sdpa: key/value length %d is shorter than query length %d",
			keyValueLength,
			queryLength,
		)
	}

	keyValueHeadStride := keyValueLength * headDim
	keyValueBatchStride := numHeads * keyValueHeadStride
	stateDict.EnsureOperationOutLen(queryTotal)

	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			queryOffset := b*queryBatchStride + h*queryHeadStride
			keyValueOffset := b*keyValueBatchStride + h*keyValueHeadStride
			outHead := stateDict.Out[queryOffset : queryOffset+queryHeadStride]
			queryHead := Q[queryOffset : queryOffset+queryHeadStride]
			keyHead := K[keyValueOffset : keyValueOffset+keyValueHeadStride]
			valueHead := V[keyValueOffset : keyValueOffset+keyValueHeadStride]

			if stateDict.Causal {
				sdpaHeadCausal(
					outHead,
					queryHead,
					keyHead,
					valueHead,
					queryLength,
					keyValueLength,
					headDim,
				)
				continue
			}

			sdpaHead(outHead, queryHead, keyHead, valueHead, queryLength, keyValueLength, headDim)
		}
	}

	return stateDict, nil
}

func attentionShape4(name string, stateDict *state.Dict) (int, int, int, int, error) {
	shape := stateDict.OperationShape()

	if len(shape) != 4 {
		return 0, 0, 0, 0, fmt.Errorf("%s: expected rank 4, got %d", name, len(shape))
	}

	for index, dimension := range shape {
		if dimension <= 0 {
			return 0, 0, 0, 0, fmt.Errorf(
				"%s: shape[%d] must be positive, got %d",
				name, index, dimension,
			)
		}
	}

	return shape[0], shape[1], shape[2], shape[3], nil
}
