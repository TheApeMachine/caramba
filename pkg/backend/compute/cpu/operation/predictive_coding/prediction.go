package predictive_coding

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Prediction computes the top-down prediction μ̂ = W @ r from a higher-level
representation r through generative weights W. This is the forward pass of the
generative model in a predictive coding hierarchy.
shape=[D_out, D_in], data[0]=W [D_out*D_in], data[1]=r [D_in] → μ̂ [D_out].
*/
type Prediction struct{}

/*
NewPrediction instantiates a new Prediction operation.
*/
func NewPrediction() *Prediction { return &Prediction{} }

/*
Forward computes μ̂ = W @ r where W is [D_out, D_in] row-major and r is [D_in].
*/
func (op *Prediction) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("predictive_coding.prediction", 2); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("predictive_coding.prediction: expected rank >= 2, got %d", len(shape))
	}

	dOut, dIn := shape[0], shape[1]
	needW := rowMajorWeightLen(dOut, dIn)

	if len(stateDict.Inputs[0]) != needW {
		return nil, fmt.Errorf(
			"predictive_coding.prediction: W length %d does not match D_out*D_in=%d",
			len(stateDict.Inputs[0]), needW,
		)
	}

	if len(stateDict.Inputs[1]) != dIn {
		return nil, fmt.Errorf(
			"predictive_coding.prediction: r length %d does not match D_in=%d",
			len(stateDict.Inputs[1]), dIn,
		)
	}

	stateDict.EnsureOperationOutLen(dOut)
	applyMatVec(stateDict.Out, stateDict.Inputs[0], stateDict.Inputs[1], dOut, dIn)

	return stateDict, nil
}
