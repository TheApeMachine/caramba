package math

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
LayerNorm normalizes over the last dimension.
data[0]=x [batch*seq*d_model], shape=[batch, seq, d_model].
y = (x - mean) / sqrt(var + eps) * Weight + Bias.
Each row runs through a fused AVX2/SSE2/NEON kernel: mean, variance,
sqrt, normalize, and affine transform all in one assembly pass.
*/
type LayerNorm struct{}

func NewLayerNorm() *LayerNorm {
	return &LayerNorm{}
}

func (layerNorm *LayerNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.layernorm"); err != nil {
		return nil, err
	}

	dModel := stateDict.OperationLastDim()

	if dModel <= 0 {
		return nil, fmt.Errorf("math.layernorm: last dimension must be positive, got %d", dModel)
	}

	if len(stateDict.Inputs[0])%dModel != 0 {
		return nil, fmt.Errorf(
			"math.layernorm: input length %d is not divisible by dim %d",
			len(stateDict.Inputs[0]), dModel,
		)
	}

	if len(stateDict.Weight) != dModel {
		return nil, fmt.Errorf(
			"math.layernorm: weight length %d does not match dim %d",
			len(stateDict.Weight), dModel,
		)
	}

	if len(stateDict.Bias) != dModel {
		return nil, fmt.Errorf(
			"math.layernorm: bias length %d does not match dim %d",
			len(stateDict.Bias), dModel,
		)
	}

	layerNormKernel(
		stateDict.Out, stateDict.Inputs[0], stateDict.Weight, stateDict.Bias,
		stateDict.Eps, dModel,
	)

	return stateDict, nil
}
