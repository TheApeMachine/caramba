package active_inference

import "fmt"

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
Forward computes the expected free energy G[k] = -sum_i q[i,k]*ln(q[i,k]) for each outcome k.
*/
func (op *ExpectedFreeEnergy) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Sprintf("active_inference: ExpectedFreeEnergy.Forward: len(shape)=%d, need >= 2", len(shape)))
	}

	n, k := shape[0], shape[1]

	if len(data) < 1 {
		panic(fmt.Sprintf("active_inference: ExpectedFreeEnergy.Forward: len(data)=%d, need >= 1", len(data)))
	}

	if len(data[0]) != n*k {
		panic(fmt.Sprintf(
			"active_inference: ExpectedFreeEnergy.Forward: len(q_outcomes)=%d, need N*K=%d",
			len(data[0]), n*k,
		))
	}

	out := make([]float64, k)
	applyExpectedFreeEnergy(out, data[0], n, k)

	return out
}
