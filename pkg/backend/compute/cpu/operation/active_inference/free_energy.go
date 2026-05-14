package active_inference

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
FreeEnergy computes the variational free energy under a Gaussian approximate
posterior, implementing Karl Friston's Free Energy Principle.

Let log_var denote the log-variance (natural log of variance sigma^2), so data[1] is log_var [N].

	F = 0.5 * sum(mu^2 + exp(log_var) - log_var - 1)

shape = [N], data[0] = mu [N], data[1] = log_var [N] → scalar [1].
*/
type FreeEnergy struct{}

/*
NewFreeEnergy instantiates a new FreeEnergy operation.
*/
func NewFreeEnergy() *FreeEnergy { return &FreeEnergy{} }

/*
Forward computes the Gaussian KL free energy
F = 0.5 * sum(mu^2 + exp(log_var) - log_var - 1) with data[1] holding log_var (log of variance).
*/
func (freeEnergy *FreeEnergy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("active_inference.free_energy: len(shape)=%d, need >= 1", len(shape))
	}

	observations := shape[0]

	if err := stateDict.RequireOperationInputs("active_inference.free_energy", 2); err != nil {
		return nil, err
	}

	if observations == 0 {
		if len(stateDict.Inputs[0]) != 0 || len(stateDict.Inputs[1]) != 0 {
			return nil, fmt.Errorf(
				"active_inference.free_energy: N=0 requires empty mu and log_var (got len %d, %d)",
				len(stateDict.Inputs[0]), len(stateDict.Inputs[1]),
			)
		}

		stateDict.SetOperationOutput([]float64{0})

		return stateDict, nil
	}

	if len(stateDict.Inputs[0]) != observations {
		return nil, fmt.Errorf(
			"active_inference.free_energy: len(mu)=%d, need N=%d",
			len(stateDict.Inputs[0]), observations,
		)
	}

	if len(stateDict.Inputs[1]) != observations {
		return nil, fmt.Errorf(
			"active_inference.free_energy: len(log_var)=%d, need N=%d",
			len(stateDict.Inputs[1]), observations,
		)
	}

	stateDict.SetOperationOutput([]float64{
		applyFreeEnergy(stateDict.Inputs[0], stateDict.Inputs[1], observations, nil),
	})

	return stateDict, nil
}
