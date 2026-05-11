package active_inference

import "fmt"

/*
FreeEnergy computes the variational free energy under a Gaussian approximate
posterior, implementing Karl Friston's Free Energy Principle.

For a Gaussian q with mean mu and log-variance log_sigma^2:
  F = 0.5 * sum(mu^2 + exp(log_sigma^2) - log_sigma^2 - 1)

shape = [N], data[0] = mu [N], data[1] = log_sigma [N] → scalar [1].
*/
type FreeEnergy struct{}

/*
NewFreeEnergy instantiates a new FreeEnergy operation.
*/
func NewFreeEnergy() *FreeEnergy { return &FreeEnergy{} }

/*
Forward computes the Gaussian KL-divergence free energy F = 0.5*sum(mu^2 + sigma^2 - ln(sigma^2) - 1).
*/
func (op *FreeEnergy) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 {
		panic(fmt.Sprintf("active_inference: FreeEnergy.Forward: len(shape)=%d, need >= 1", len(shape)))
	}

	n := shape[0]

	if len(data) < 2 {
		panic(fmt.Sprintf("active_inference: FreeEnergy.Forward: len(data)=%d, need >= 2", len(data)))
	}

	if len(data[0]) != n {
		panic(fmt.Sprintf("active_inference: FreeEnergy.Forward: len(mu)=%d, need N=%d", len(data[0]), n))
	}

	if len(data[1]) != n {
		panic(fmt.Sprintf("active_inference: FreeEnergy.Forward: len(log_sigma)=%d, need N=%d", len(data[1]), n))
	}

	return []float64{applyFreeEnergy(data[0], data[1], n)}
}
