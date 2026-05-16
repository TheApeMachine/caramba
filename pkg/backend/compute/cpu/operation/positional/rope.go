package positional

import (
	"fmt"
	"math"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/rotary"
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

const (
	ropeModeAdjacent = "adjacent"
	ropeModeHalf     = "half"
)

/*
NewRoPE instantiates a stateless RoPE operation.
*/
func NewRoPE(args ...any) *RoPE {
	return &RoPE{}
}

func buildRoPETables(
	config rotary.Config,
	headDim int,
	seqLen int,
	positionStart int,
) (cosTable, sinTable []float64, err error) {
	numPairs := headDim / 2
	n := seqLen * numPairs
	cosTable = make([]float64, n)
	sinTable = make([]float64, n)

	if seqLen == 0 || numPairs == 0 {
		return cosTable, sinTable, nil
	}

	frequencies, err := config.InverseFrequencies(headDim)

	if err != nil {
		return nil, nil, err
	}

	cosStep := make([]float64, numPairs)
	sinStep := make([]float64, numPairs)

	for i := 0; i < numPairs; i++ {
		theta := frequencies[i]
		cosStep[i] = math.Cos(theta)
		sinStep[i] = math.Sin(theta)
		angle := float64(positionStart) * theta
		cosTable[i] = math.Cos(angle)
		sinTable[i] = math.Sin(angle)
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

func ropeMode(mode string) (string, error) {
	switch strings.ToLower(mode) {
	case "", ropeModeAdjacent:
		return ropeModeAdjacent, nil
	case ropeModeHalf:
		return ropeModeHalf, nil
	default:
		return "", fmt.Errorf("positional.rope: unsupported mode %q", mode)
	}
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
	cosTable, sinTable, err := buildRoPETables(
		rotary.Config{
			Base:                          stateDict.Base,
			Type:                          stateDict.RoPEType,
			Factor:                        stateDict.RoPEFactor,
			LowFreqFactor:                 stateDict.RoPELowFreqFactor,
			HighFreqFactor:                stateDict.RoPEHighFreqFactor,
			OriginalMaxPositionEmbeddings: stateDict.RoPEOriginalContext,
		},
		headDim,
		seqLen,
		stateDict.PositionStart,
	)

	if err != nil {
		return nil, err
	}

	mode, err := ropeMode(stateDict.Mode)

	if err != nil {
		return nil, err
	}

	if mode == ropeModeHalf {
		ropeKernelHalf(
			stateDict.Out, stateDict.Inputs[0], cosTable, sinTable,
			batch, numHeads, seqLen, numPairs,
		)

		return stateDict, nil
	}

	ropeKernel(
		stateDict.Out, stateDict.Inputs[0], cosTable, sinTable,
		batch, numHeads, seqLen, numPairs,
	)

	return stateDict, nil
}
