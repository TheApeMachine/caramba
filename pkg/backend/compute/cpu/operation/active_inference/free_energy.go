package active_inference

import "fmt"

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
func (op *FreeEnergy) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 {
		panic(fmt.Sprintf("active_inference: FreeEnergy.Forward: len(shape)=%d, need >= 1", len(shape)))
	}

	n := shape[0]

	if len(data) < 2 {
		panic(fmt.Sprintf("active_inference: FreeEnergy.Forward: len(data)=%d, need >= 2", len(data)))
	}

	if n == 0 {
		if len(data[0]) != 0 || len(data[1]) != 0 {
			panic(fmt.Sprintf(
				"active_inference: FreeEnergy.Forward: N=0 requires empty mu and log_var (got len %d, %d)",
				len(data[0]), len(data[1]),
			))
		}

		return []float64{0}
	}

	if len(data[0]) != n {
		panic(fmt.Sprintf("active_inference: FreeEnergy.Forward: len(mu)=%d, need N=%d", len(data[0]), n))
	}

	if len(data[1]) != n {
		panic(fmt.Sprintf("active_inference: FreeEnergy.Forward: len(log_var)=%d, need N=%d", len(data[1]), n))
	}

	return []float64{applyFreeEnergy(data[0], data[1], n, nil)}
}
