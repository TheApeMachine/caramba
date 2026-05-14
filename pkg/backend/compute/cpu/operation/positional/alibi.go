package positional

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
ALiBi computes the Attention with Linear Biases tensor.

For each head h, query position q, and key position k:

	bias[h, q, k] = slopes[h] * (k - q)   (causal)
	bias[h, q, k] = slopes[h] * |k - q|   (non-causal, i.e. abs)

Slope construction follows the paper (Press et al., 2022):
  - For num_heads that is a power of 2:
    slopes[i] = 2^( -(8/num_heads) * (i+1) )
  - For other num_heads: use slopes for the next power of 2,
    then interleave with slopes for twice that power.

Forward:
  - shape=[num_heads, seq_len_q, seq_len_k]
  - no data required (pass nil or empty)
  - output: [num_heads * seq_len_q * seq_len_k] bias values (row-major)
*/
type ALiBi struct{}

// NewALiBi returns an ALiBi for the given number of heads.
func NewALiBi(args ...any) *ALiBi {
	return &ALiBi{}
}

// buildSlopes returns the per-head slope values.
func buildSlopes(numHeads int) []float64 {
	// next power of 2 >= numHeads
	n := 1
	for n < numHeads {
		n <<= 1
	}
	pow2Slopes := func(k int) []float64 {
		s := make([]float64, k)
		step := 8.0 / float64(k)
		for i := range s {
			s[i] = math.Pow(2, -step*float64(i+1))
		}
		return s
	}
	if n == numHeads {
		return pow2Slopes(n)
	}
	// non-power-of-2: interleave n/2 slopes and n slopes
	half := pow2Slopes(n / 2)
	full := pow2Slopes(n)
	slopes := make([]float64, numHeads)
	hi, fi := 0, 0
	for i := 0; i < numHeads; i++ {
		if i%2 == 0 && hi < len(half) {
			slopes[i] = half[hi]
			hi++
		} else if fi < len(full) {
			slopes[i] = full[fi]
			fi++
		}
	}
	return slopes
}

// Forward computes the ALiBi bias tensor.
// shape=[num_heads, seq_len_q, seq_len_k]; data is ignored.
func (alibi *ALiBi) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) != 3 {
		return nil, fmt.Errorf("positional.alibi: expected rank 3, got %d", len(shape))
	}

	numHeads := shape[0]
	seqLenQ := shape[1]
	seqLenK := shape[2]

	slopes := buildSlopes(numHeads)
	stateDict.EnsureOperationOutLen(numHeads * seqLenQ * seqLenK)
	alibiKernel(stateDict.Out, slopes, seqLenQ, seqLenK, stateDict.Causal)

	return stateDict, nil
}
