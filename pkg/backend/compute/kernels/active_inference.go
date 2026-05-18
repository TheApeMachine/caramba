package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
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
*/

func init() {
	Default.Register(Kernel{
		Name: "free_energy",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runFreeEnergy,
	})

	Default.Register(Kernel{
		Name: "expected_free_energy",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runExpectedFreeEnergy,
	})

	Default.Register(Kernel{
		Name: "belief_update",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runBeliefUpdate,
	})

	Default.Register(Kernel{
		Name: "precision_weight",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runPrecisionWeight,
	})
}

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

	const eps = 1e-12

	var crossEntropy, kl float64

	for index, posteriorValue := range posterior {
		clampedLike := math.Max(eps, float64(likelihood[index]))
		clampedPosterior := math.Max(eps, float64(posteriorValue))
		clampedPrior := math.Max(eps, float64(prior[index]))

		crossEntropy += -float64(posteriorValue) * math.Log(clampedLike)
		kl += float64(posteriorValue) * (math.Log(clampedPosterior) - math.Log(clampedPrior))
	}

	out[0] = float32(crossEntropy + kl)
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

	const eps = 1e-12

	var pragmatic, epistemic float64

	for index, predicted := range predictedObs {
		predictedClamped := math.Max(eps, float64(predicted))
		preferredClamped := math.Max(eps, float64(preferredObs[index]))

		pragmatic += float64(predicted) * (math.Log(predictedClamped) - math.Log(preferredClamped))
	}

	for _, stateValue := range predictedState {
		clamped := math.Max(eps, float64(stateValue))
		epistemic += -float64(stateValue) * math.Log(clamped)
	}

	out[0] = float32(pragmatic + epistemic)
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

	var sum float64

	for index, likeValue := range likelihood {
		product := likeValue * prior[index]
		out[index] = product
		sum += float64(product)
	}

	if sum == 0 {
		return nil
	}

	normalizer := 1.0 / float32(sum)

	for index := range out {
		out[index] *= normalizer
	}

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

	for index, value := range errors {
		out[index] = value * precision[index]
	}

	return nil
}
