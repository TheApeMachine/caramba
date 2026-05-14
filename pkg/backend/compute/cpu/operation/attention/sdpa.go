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

	batch, numHeads, seqLen, headDim, err := attentionShape4("attention.sdpa", stateDict)

	if err != nil {
		return nil, err
	}

	Q, K, V := stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2]
	headStride := seqLen * headDim
	batchStride := numHeads * headStride
	total := batch * numHeads * headStride

	if len(Q) != total || len(K) != total || len(V) != total {
		return nil, fmt.Errorf("attention.sdpa: Q/K/V lengths must all equal %d", total)
	}

	stateDict.EnsureOperationOutLen(total)

	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			off := b*batchStride + h*headStride
			sdpaHead(
				stateDict.Out[off:off+headStride],
				Q[off:off+headStride],
				K[off:off+headStride],
				V[off:off+headStride],
				seqLen, headDim, nil,
			)
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
