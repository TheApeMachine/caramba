package attention

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// SlidingWindow implements sliding-window attention (local attention).
// Each position i can only attend to positions in [i-Window, i].
//
// shape: [batch, num_heads, seq_len, head_dim]
// data[0]=Q, data[1]=K, data[2]=V  each [batch*num_heads*seq_len*head_dim]
// output: [batch*num_heads*seq_len*head_dim]
type SlidingWindow struct {
}

func NewSlidingWindow(window ...int) *SlidingWindow {
	return &SlidingWindow{}
}

func (sw *SlidingWindow) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("attention.sliding_window", 3); err != nil {
		return nil, err
	}

	batch, numHeads, seqLen, headDim, err := attentionShape4("attention.sliding_window", stateDict)

	if err != nil {
		return nil, err
	}

	Q, K, V := stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2]
	headStride := seqLen * headDim
	batchStride := numHeads * headStride
	total := batch * numHeads * headStride

	if len(Q) != total || len(K) != total || len(V) != total {
		return nil, fmt.Errorf("attention.sliding_window: Q/K/V lengths must all equal %d", total)
	}

	if stateDict.Window < 0 {
		return nil, fmt.Errorf("attention.sliding_window: window must be non-negative")
	}

	stateDict.EnsureOperationOutLen(total)

	window := stateDict.Window
	maskFn := func(i, j int) bool {
		return j > i || j < i-window
	}

	for b := range batch {
		for h := range numHeads {
			off := b*batchStride + h*headStride
			sdpaHeadWithInf(
				stateDict.Out[off:off+headStride],
				Q[off:off+headStride],
				K[off:off+headStride],
				V[off:off+headStride],
				seqLen, headDim, maskFn,
			)
		}
	}

	return stateDict, nil
}

// sdpaHeadWithInf is like sdpaHead but supports -inf masking.
func sdpaHeadWithInf(out, q, k, v []float64, seqLen, headDim int, maskFn func(i, j int) bool) {
	scale := 1.0 / math.Sqrt(float64(headDim))
	scores := make([]float64, seqLen)
	validMask := make([]float64, seqLen)

	for i := range seqLen {
		qRow := q[i*headDim : (i+1)*headDim]

		for j := range seqLen {
			if maskFn(i, j) {
				scores[j] = math.Inf(-1)
				validMask[j] = 0
				continue
			}

			kRow := k[j*headDim : (j+1)*headDim]
			scores[j] = dotProduct(qRow, kRow) * scale
			validMask[j] = 1
		}

		// finite max over valid scores
		mx := math.Inf(-1)
		for j, s := range scores {
			if validMask[j] != 0 && s > mx {
				mx = s
			}
		}

		// scores[j] := exp(scores[j] - max) for valid; 0 for masked
		// First zero out -Inf entries so ExpVec doesn't get NaN, then shift.
		for j, s := range scores {
			if validMask[j] == 0 {
				scores[j] = 0
			} else {
				scores[j] = s - mx
			}
		}

		for scoreIndex := range scores {
			scores[scoreIndex] = math.Exp(scores[scoreIndex]) * validMask[scoreIndex]
		}

		sum := 0.0

		for _, score := range scores {
			sum += score
		}

		if sum > 0 {
			scale := 1.0 / sum

			for scoreIndex := range scores {
				scores[scoreIndex] *= scale
			}
		}

		outRow := out[i*headDim : (i+1)*headDim]

		for d := range outRow {
			outRow[d] = 0
		}

		for j := range seqLen {
			vRow := v[j*headDim : (j+1)*headDim]
			scaledAdd(outRow, vRow, scores[j])
		}
	}
}
