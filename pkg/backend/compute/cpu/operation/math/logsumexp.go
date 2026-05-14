package math

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
LogSumExp computes log(sum(exp(x))) over the last dimension via a dedicated
AVX2/SSE2/NEON kernel: max, exp, sum, log all fused inline.
*/
type LogSumExp struct{}

func NewLogSumExp() *LogSumExp { return &LogSumExp{} }

func (logSumExp *LogSumExp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.logsumexp"); err != nil {
		return nil, err
	}

	dimSize := stateDict.OperationLastDim()

	if dimSize <= 0 {
		return nil, fmt.Errorf("math.logsumexp: last dimension must be positive, got %d", dimSize)
	}

	if len(stateDict.Inputs[0])%dimSize != 0 {
		return nil, fmt.Errorf(
			"math.logsumexp: input length %d is not divisible by dim %d",
			len(stateDict.Inputs[0]), dimSize,
		)
	}

	stateDict.EnsureOperationOutLen(len(stateDict.Inputs[0]) / dimSize)
	logSumExpKernel(stateDict.Out, stateDict.Inputs[0], dimSize)

	return stateDict, nil
}
