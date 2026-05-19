package active_inference

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func RunFreeEnergy(args ...tensor.Tensor) error {
	return runFreeEnergy(args...)
}

func RunExpectedFreeEnergy(args ...tensor.Tensor) error {
	return runExpectedFreeEnergy(args...)
}

func RunBeliefUpdate(args ...tensor.Tensor) error {
	return runBeliefUpdate(args...)
}

func RunPrecisionWeight(args ...tensor.Tensor) error {
	return runPrecisionWeight(args...)
}
