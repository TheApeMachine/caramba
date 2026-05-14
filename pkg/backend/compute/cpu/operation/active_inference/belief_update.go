package active_inference

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
BeliefUpdate performs a gradient descent step on the variational free energy
to update Gaussian belief parameters mu and log_sigma.

Gradients of F w.r.t. parameters:

	dF/dmu        = mu + prediction_error
	dF/dlog_sigma = exp(log_sigma) - 1   (from the KL term)

Update rules:

	mu_new        = mu        - lr * (mu + prediction_error)
	log_sigma_new = log_sigma - lr * (exp(log_sigma) - 1)

shape must have len(shape) >= 2: shape[0] = N (batch / belief dimension), shape[1] = lr
encoded as an integer step count; the effective learning rate is

	lr := float64(shape[1]) * 1e-4

with 0 < lr <= 1 (i.e. 1 <= shape[1] <= 10000).

data[0] = mu [N], data[1] = log_sigma [N], data[2] = prediction_error [N]

Forward returns a vector of length 2*N: indices [0:N) are updated mu, [N:2*N) are updated log_sigma.
*/
type BeliefUpdate struct{}

/*
NewBeliefUpdate instantiates a new BeliefUpdate operation.
*/
func NewBeliefUpdate() *BeliefUpdate { return &BeliefUpdate{} }

/*
Forward computes one gradient descent step updating mu and log_sigma.
Requires shape[0] = N > 0 and shape[1] so lr = float64(shape[1])*1e-4 satisfies 0 < lr <= 1.
Returns [mu_new || log_sigma_new] of length 2*N.
*/
func (beliefUpdate *BeliefUpdate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("active_inference.belief_update: len(shape)=%d, need >= 2", len(shape))
	}

	dimension := shape[0]

	if dimension <= 0 {
		return nil, fmt.Errorf(
			"active_inference.belief_update: dimension=%d from shape[0] must be positive",
			dimension,
		)
	}

	lrStep := shape[1]
	lr := float64(lrStep) * 1e-4

	if lrStep <= 0 {
		return nil, fmt.Errorf(
			"active_inference.belief_update: shape[1]=%d yields non-positive lr=%g (need shape[1] >= 1)",
			lrStep, lr,
		)
	}

	if lr > 1.0 {
		return nil, fmt.Errorf(
			"active_inference.belief_update: shape[1]=%d yields lr=%g > 1 (cap at shape[1]=10000)",
			lrStep, lr,
		)
	}

	if err := stateDict.RequireOperationInputs("active_inference.belief_update", 3); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs[0]) != dimension {
		return nil, fmt.Errorf(
			"active_inference.belief_update: len(mu)=%d, need N=%d",
			len(stateDict.Inputs[0]), dimension,
		)
	}

	if len(stateDict.Inputs[1]) != dimension {
		return nil, fmt.Errorf(
			"active_inference.belief_update: len(log_sigma)=%d, need N=%d",
			len(stateDict.Inputs[1]), dimension,
		)
	}

	if len(stateDict.Inputs[2]) != dimension {
		return nil, fmt.Errorf(
			"active_inference.belief_update: len(pred_err)=%d, need N=%d",
			len(stateDict.Inputs[2]), dimension,
		)
	}

	stateDict.EnsureOperationOutLen(2 * dimension)
	applyBeliefUpdate(
		stateDict.Out[:dimension],
		stateDict.Out[dimension:],
		stateDict.Inputs[0],
		stateDict.Inputs[1],
		stateDict.Inputs[2],
		lr,
		dimension,
	)

	return stateDict, nil
}
