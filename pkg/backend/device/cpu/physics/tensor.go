package physics

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func RunLaplacian(args ...tensor.Tensor) error {
	return runLaplacian(args...)
}

func RunLaplacian4(args ...tensor.Tensor) error {
	return runLaplacian4(args...)
}

func RunGrad1D(args ...tensor.Tensor) error {
	return runGrad1D(args...)
}

func RunDivergence1D(args ...tensor.Tensor) error {
	return runDivergence1D(args...)
}

func RunFFT1DDefault(args ...tensor.Tensor) error {
	return runFFT1DDefault(args...)
}

func RunIFFT1DDefault(args ...tensor.Tensor) error {
	return runIFFT1DDefault(args...)
}

func RunQuantumPotential(args ...tensor.Tensor) error {
	return runQuantumPotential(args...)
}

func RunBohmianVelocity(args ...tensor.Tensor) error {
	return runBohmianVelocity(args...)
}

func RunMadelungContinuity(args ...tensor.Tensor) error {
	return runMadelungContinuity(args...)
}
