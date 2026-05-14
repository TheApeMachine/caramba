package math

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
RMSNorm computes y = x / rms(x) * Weight where rms = sqrt(mean(x²) + eps).
Each row is dispatched to a fused AVX2/SSE2/NEON kernel.
*/
type RMSNorm struct{}

func NewRMSNorm() *RMSNorm {
	return &RMSNorm{}
}

func (rmsNorm *RMSNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.rmsnorm"); err != nil {
		return nil, err
	}

	dModel := stateDict.OperationLastDim()

	if dModel <= 0 {
		return nil, fmt.Errorf("math.rmsnorm: last dimension must be positive, got %d", dModel)
	}

	if len(stateDict.Inputs[0])%dModel != 0 {
		return nil, fmt.Errorf(
			"math.rmsnorm: input length %d is not divisible by dim %d",
			len(stateDict.Inputs[0]), dModel,
		)
	}

	if len(stateDict.Weight) != dModel {
		return nil, fmt.Errorf(
			"math.rmsnorm: weight length %d does not match dim %d",
			len(stateDict.Weight), dModel,
		)
	}

	rmsNormKernel(stateDict.Out, stateDict.Inputs[0], stateDict.Weight, stateDict.Eps, dModel)

	return stateDict, nil
}
