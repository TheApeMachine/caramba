package active_inference

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
PrecisionWeight scales prediction errors by precision (inverse variance),
implementing the precision-weighted prediction error central to Active Inference.

	out[i] = error[i] * exp(log_precision[i])

shape = [N], data[0] = error [N], data[1] = log_precision [N] → weighted error [N].
*/
type PrecisionWeight struct{}

/*
NewPrecisionWeight instantiates a new PrecisionWeight operation.
*/
func NewPrecisionWeight() *PrecisionWeight { return &PrecisionWeight{} }

/*
Forward computes precision-weighted prediction errors: out[i] = error[i] * exp(log_prec[i]).
Shape must be exactly one-dimensional: len(shape) == 1 with shape[0] = N. N may be zero (returns empty out).
*/
func (precisionWeight *PrecisionWeight) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) != 1 {
		return nil, fmt.Errorf(
			"active_inference.precision_weight: len(shape)=%d, need exactly 1",
			len(shape),
		)
	}

	dimension := shape[0]

	if err := stateDict.RequireOperationInputs("active_inference.precision_weight", 2); err != nil {
		return nil, err
	}

	if dimension == 0 {
		stateDict.SetOperationOutput([]float64{})

		return stateDict, nil
	}

	if len(stateDict.Inputs[0]) != dimension {
		return nil, fmt.Errorf(
			"active_inference.precision_weight: len(error)=%d, need N=%d",
			len(stateDict.Inputs[0]), dimension,
		)
	}

	if len(stateDict.Inputs[1]) != dimension {
		return nil, fmt.Errorf(
			"active_inference.precision_weight: len(log_precision)=%d, need N=%d",
			len(stateDict.Inputs[1]), dimension,
		)
	}

	stateDict.EnsureOperationOutLen(dimension)
	applyPrecisionWeight(
		stateDict.Out,
		stateDict.Inputs[0],
		stateDict.Inputs[1],
		dimension,
		nil,
	)

	return stateDict, nil
}
