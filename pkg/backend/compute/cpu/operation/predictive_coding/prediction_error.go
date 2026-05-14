package predictive_coding

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
PredictionError computes the precision-weighted prediction error ε = Π ⊙ (x - μ̂)
where Π is the precision (inverse variance) diagonal. When no precision is provided
(data has only 2 elements), plain error x - μ̂ is returned.
shape=[N], data[0]=x [N], data[1]=mu_hat [N], data[2]=precision [N] (optional) → ε [N].
*/
type PredictionError struct{}

/*
NewPredictionError instantiates a new PredictionError operation.
*/
func NewPredictionError() *PredictionError { return &PredictionError{} }

/*
Forward computes ε = Π ⊙ (x - μ̂).
*/
func (op *PredictionError) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("predictive_coding.prediction_error", 2); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("predictive_coding.prediction_error: shape is required")
	}

	n := shape[0]

	if n <= 0 {
		return nil, fmt.Errorf("predictive_coding.prediction_error: n must be positive, got %d", n)
	}

	if len(stateDict.Inputs[0]) != n || len(stateDict.Inputs[1]) != n {
		return nil, fmt.Errorf("predictive_coding.prediction_error: x and mu_hat must match n")
	}

	stateDict.EnsureOperationOutLen(n)
	applySubVec(stateDict.Out, stateDict.Inputs[0], stateDict.Inputs[1])

	if len(stateDict.Inputs) >= 3 {
		if len(stateDict.Inputs[2]) != n {
			return nil, fmt.Errorf("predictive_coding.prediction_error: precision length must match n")
		}

		applyMulVec(stateDict.Out, stateDict.Out, stateDict.Inputs[2])
	}

	return stateDict, nil
}
