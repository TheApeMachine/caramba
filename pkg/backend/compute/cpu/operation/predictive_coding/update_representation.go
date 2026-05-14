package predictive_coding

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
UpdateRepresentation performs the representation update step of predictive coding:
r += lr * (W^T @ ε_lower - ε_self)
where W^T @ ε_lower is the bottom-up error signal propagated through transposed
generative weights, and ε_self is the prediction error from the layer above.
shape=[D_in, D_out], data[0]=r [D_in], data[1]=W [D_out*D_in],
data[2]=eps_lower [D_out], data[3]=eps_self [D_in], data[4]=lr [1] → r_new [D_in].
*/
type UpdateRepresentation struct{}

/*
NewUpdateRepresentation instantiates a new UpdateRepresentation operation.
*/
func NewUpdateRepresentation() *UpdateRepresentation { return &UpdateRepresentation{} }

/*
Forward computes r_new = r + lr * (W^T @ eps_lower - eps_self).
*/
func (op *UpdateRepresentation) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("predictive_coding.update_representation", 4); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("predictive_coding.update_representation: expected rank >= 2, got %d", len(shape))
	}

	dIn, dOut := shape[0], shape[1]

	r, W, epsLower, epsSelf := stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2], stateDict.Inputs[3]

	needW := rowMajorWeightLen(dOut, dIn)

	if len(r) != dIn || len(W) != needW || len(epsLower) != dOut || len(epsSelf) != dIn {
		return nil, fmt.Errorf(
			"predictive_coding.update_representation: shape mismatch r=%d W=%d eps_lower=%d eps_self=%d",
			len(r), len(W), len(epsLower), len(epsSelf),
		)
	}

	if stateDict.LR == 0 {
		return nil, fmt.Errorf("predictive_coding.update_representation: lr must be non-zero")
	}

	signal := make([]float64, dIn)
	applyMatVecTranspose(signal, W, epsLower, dOut, dIn)
	applySubVecInPlace(signal, epsSelf)
	stateDict.EnsureOperationOutLen(dIn)
	copy(stateDict.Out, r)
	applyAxpy(stateDict.Out, signal, stateDict.LR)

	return stateDict, nil
}
