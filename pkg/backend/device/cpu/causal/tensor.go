package causal

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func RunCholesky(args ...tensor.Tensor) error {
	return runCholesky(args...)
}

func RunBackdoorAdjustment(args ...tensor.Tensor) error {
	return runBackdoorAdjustment(args...)
}

func RunFrontdoorAdjustment(args ...tensor.Tensor) error {
	return runFrontdoorAdjustment(args...)
}

func RunDoIntervene(args ...tensor.Tensor) error {
	return runDoIntervene(args...)
}

func RunCATE(args ...tensor.Tensor) error {
	return runCATE(args...)
}

func RunCounterfactual(args ...tensor.Tensor) error {
	return runCounterfactual(args...)
}

func RunIVEstimate(args ...tensor.Tensor) error {
	return runIVEstimate(args...)
}

func RunDAGMarkovFactorization(args ...tensor.Tensor) error {
	return runDAGMarkovFactorization(args...)
}

func RunMarkovFlowActive(args ...tensor.Tensor) error {
	return runMarkovFlowActive(args...)
}

func RunMarkovFlowInternal(args ...tensor.Tensor) error {
	return runMarkovFlowInternal(args...)
}
