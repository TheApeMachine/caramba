package positional

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
RoPE applies Rotary Position Embeddings to Q or K tensors.

For each position t and each pair of dimensions (2i, 2i+1):

	theta_i = 1 / (base^(2i/head_dim))
	angle   = t * theta_i
	[x_{2i}, x_{2i+1}] -> [x_{2i}*cos(angle) - x_{2i+1}*sin(angle),
	                        x_{2i}*sin(angle) + x_{2i+1}*cos(angle)]

Forward: shape=[batch, num_heads, seq_len, head_dim], data[0]=input tensor.

The cos/sin tables are built via an angle-recurrence kernel: theta_i and the
single-step cos/sin per pair are precomputed once (O(numPairs)), then a
SIMD AVX2/SSE2/NEON kernel advances all numPairs angles per t in lockstep.
*/
type RoPE struct{}

/*
NewRoPE instantiates a stateless RoPE operation.
*/
func NewRoPE(args ...any) *RoPE {
	return &RoPE{}
}

func buildRoPETables(base float64, headDim, seqLen int) (cosTable, sinTable []float64) {
	if base == 0 {
		base = 10000.0
	}

	numPairs := headDim / 2
	n := seqLen * numPairs
	cosTable = make([]float64, n)
	sinTable = make([]float64, n)

	if seqLen == 0 || numPairs == 0 {
		return
	}

	cosStep := make([]float64, numPairs)
	sinStep := make([]float64, numPairs)

	for i := 0; i < numPairs; i++ {
		theta := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
		cosStep[i] = math.Cos(theta)
		sinStep[i] = math.Sin(theta)
	}

	for i := 0; i < numPairs; i++ {
		cosTable[i] = 1.0
		sinTable[i] = 0.0
	}

	cosCur := make([]float64, numPairs)
	sinCur := make([]float64, numPairs)
	copy(cosCur, cosTable[:numPairs])
	copy(sinCur, sinTable[:numPairs])

	for t := 1; t < seqLen; t++ {
		ropeAdvanceRow(cosCur, sinCur, cosStep, sinStep)
		copy(cosTable[t*numPairs:(t+1)*numPairs], cosCur)
		copy(sinTable[t*numPairs:(t+1)*numPairs], sinCur)
	}

	return
}

func (rope *RoPE) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("positional.rope"); err != nil {
		return nil, err
	}

	batch, numHeads, seqLen, headDim, err := stateDict.RoPELayout("positional.rope")

	if err != nil {
		return nil, err
	}

	if len(stateDict.Inputs[0]) != batch*numHeads*seqLen*headDim {
		return nil, fmt.Errorf(
			"positional.rope: input length %d does not match shape product %d",
			len(stateDict.Inputs[0]), batch*numHeads*seqLen*headDim,
		)
	}

	numPairs := headDim / 2
	cosTable, sinTable := buildRoPETables(stateDict.Base, headDim, seqLen)
	ropeKernel(
		stateDict.Out, stateDict.Inputs[0], cosTable, sinTable,
		batch, numHeads, seqLen, numPairs,
	)

	return stateDict, nil
}
