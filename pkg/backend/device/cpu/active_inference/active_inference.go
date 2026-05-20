package active_inference

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
Active inference primitives — free-energy minimization for
generative-model agents. The four kernels here cover the canonical
loop:

  - free_energy: F = -ln p(o|s) + KL[q(s) || p(s)].
  - expected_free_energy: G = epistemic + pragmatic contributions
    used by policy selection.
  - belief_update: posterior q(s|o) ∝ p(o|s) × q(s).
  - precision_weight: applies the learned precision γ to a prediction
    error tensor.

Host tensor paths route through Float32Native dispatchers; AVX-512
bodies live in f32_avx512_amd64.s and select_amd64.go.
*/

/*
runFreeEnergy computes F = E_q[-ln p(o|s)] + KL[q || p_prior] over
sample-aligned tensors. Args: (likelihood, posterior, prior, output).
The output is a scalar (length-1) free-energy value.
*/
func runFreeEnergy(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	likelihood, _ := args[0].Float32Native()
	posterior, _ := args[1].Float32Native()
	prior, _ := args[2].Float32Native()
	out, _ := args[4].Float32Native()

	if len(likelihood) != len(posterior) || len(posterior) != len(prior) ||
		len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	out[0] = FreeEnergyFloat32Native(likelihood, posterior, prior)
	return nil
}

/*
runExpectedFreeEnergy computes G = epistemic + pragmatic for a
candidate policy. Args: (predicted_obs, preferred_obs, predicted_state,
output). The epistemic term is the entropy reduction in beliefs about
hidden states; the pragmatic term is the divergence between predicted
and preferred observations.
*/
func runExpectedFreeEnergy(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	predictedObs, _ := args[0].Float32Native()
	preferredObs, _ := args[1].Float32Native()
	predictedState, _ := args[2].Float32Native()
	out, _ := args[3].Float32Native()

	if len(predictedObs) != len(preferredObs) || len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	out[0] = ExpectedFreeEnergyFloat32Native(
		predictedObs, preferredObs, predictedState,
	)
	return nil
}

/*
runBeliefUpdate computes q(s|o) ∝ p(o|s) × q_prev(s) and normalizes.
Args: (likelihood, prior, output).
*/
func runBeliefUpdate(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	likelihood, _ := args[0].Float32Native()
	prior, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(likelihood) != len(prior) || len(out) != len(prior) {
		return tensor.ErrShapeMismatch
	}

	BeliefUpdateFloat32Native(likelihood, prior, out)
	return nil
}

/*
runPrecisionWeight multiplies prediction errors by per-element
precision (inverse variance). Args: (errors, precision, output).
*/
func runPrecisionWeight(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	errors, _ := args[0].Float32Native()
	precision, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(errors) != len(precision) || len(out) != len(errors) {
		return tensor.ErrShapeMismatch
	}

	PrecisionWeightFloat32Native(errors, precision, out)
	return nil
}
