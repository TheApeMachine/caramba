package active_inference

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
ExpectedFreeEnergy computes the expected free energy G for policy selection
under the Free Energy Principle. G combines:
  - Epistemic value (information gain): H[p(o|pi)] = -sum(p_o * ln p_o)
  - Pragmatic value (KL divergence):    KL(q(s|pi) || p(s))

For K outcome categories with predicted probabilities q_outcomes[k]:

	G[k] = -sum_i(q_outcomes[i*K+k] * ln(q_outcomes[i*K+k] + eps))

shape = [N, K]: N = number of state dimensions, K = number of outcome categories.
data[0] = q_outcomes [N*K] row-major (q[i,k] = q_outcomes[i*K+k]).
→ G [K], one expected free energy value per outcome.
*/
type ExpectedFreeEnergy struct{}

/*
NewExpectedFreeEnergy instantiates a new ExpectedFreeEnergy operation.
*/
func NewExpectedFreeEnergy() *ExpectedFreeEnergy { return &ExpectedFreeEnergy{} }

/*
Forward computes G[k] = -sum_i q_outcomes[i*K+k] * ln(q_outcomes[i*K+k] + eps) for each outcome k,
with eps a small positive constant for numerical stability (see applyExpectedFreeEnergy).
*/
func (expectedFreeEnergy *ExpectedFreeEnergy) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf(
			"active_inference.expected_free_energy: len(shape)=%d, need >= 2",
			len(shape),
		)
	}

	dimensions, outcomes := shape[0], shape[1]

	if dimensions <= 0 || outcomes <= 0 {
		return nil, fmt.Errorf(
			"active_inference.expected_free_energy: need shape[0]=N > 0 and shape[1]=K > 0 (got N=%d K=%d)",
			dimensions, outcomes,
		)
	}

	if err := stateDict.RequireOperation("active_inference.expected_free_energy"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs[0]) != dimensions*outcomes {
		return nil, fmt.Errorf(
			"active_inference.expected_free_energy: len(q_outcomes)=%d, need N*K=%d",
			len(stateDict.Inputs[0]), dimensions*outcomes,
		)
	}

	for index, value := range stateDict.Inputs[0] {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf(
				"active_inference.expected_free_energy: q_outcomes[%d]=%g is not finite",
				index, value,
			)
		}
	}

	stateDict.EnsureOperationOutLen(outcomes)
	applyExpectedFreeEnergy(
		stateDict.Out,
		stateDict.Inputs[0],
		dimensions,
		outcomes,
	)

	return stateDict, nil
}
