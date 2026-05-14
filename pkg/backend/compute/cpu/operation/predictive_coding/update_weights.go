package predictive_coding

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
UpdateWeights performs the Hebbian weight update step of predictive coding:
ΔW = lr * ε @ r^T, giving W_new = W + ΔW.
The generative weights are updated to reduce future prediction errors.
shape=[D_out, D_in], data[0]=W [D_out*D_in], data[1]=eps [D_out],
data[2]=r [D_in], data[3]=lr [1] → W_new [D_out*D_in].
*/
type UpdateWeights struct{}

/*
NewUpdateWeights instantiates a new UpdateWeights operation.
*/
func NewUpdateWeights() *UpdateWeights { return &UpdateWeights{} }

/*
Forward computes W_new[i*D_in+j] = W[i*D_in+j] + lr * eps[i] * r[j].
*/
func (op *UpdateWeights) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("predictive_coding.update_weights", 3); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("predictive_coding.update_weights: expected rank >= 2, got %d", len(shape))
	}

	dOut, dIn := shape[0], shape[1]

	W, eps, r := stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2]

	needW := rowMajorWeightLen(dOut, dIn)

	if len(W) != needW || len(eps) != dOut || len(r) != dIn {
		return nil, fmt.Errorf(
			"predictive_coding.update_weights: shape mismatch W=%d eps=%d r=%d",
			len(W), len(eps), len(r),
		)
	}

	if stateDict.LR == 0 {
		return nil, fmt.Errorf("predictive_coding.update_weights: lr must be non-zero")
	}

	stateDict.EnsureOperationOutLen(dOut * dIn)
	copy(stateDict.Out, W)
	applyOuterAdd(stateDict.Out, eps, r, stateDict.LR, dOut, dIn)

	return stateDict, nil
}
