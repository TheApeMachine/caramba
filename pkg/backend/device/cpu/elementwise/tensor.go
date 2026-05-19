package elementwise

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func RunAbsFloat32(args ...tensor.Tensor) error {
	return runAbsFloat32(args...)
}

func RunNegFloat32(args ...tensor.Tensor) error {
	return runNegFloat32(args...)
}

func RunSqrtFloat32(args ...tensor.Tensor) error {
	return runSqrtFloat32(args...)
}

func RunReluFloat32(args ...tensor.Tensor) error {
	return runReluFloat32(args...)
}

func RunAbsBFloat16(args ...tensor.Tensor) error {
	return runAbsBFloat16(args...)
}

func RunNegBFloat16(args ...tensor.Tensor) error {
	return runNegBFloat16(args...)
}

func RunSqrtBFloat16(args ...tensor.Tensor) error {
	return runSqrtBFloat16(args...)
}

func RunReluBFloat16(args ...tensor.Tensor) error {
	return runReluBFloat16(args...)
}

func RunAbsFloat16(args ...tensor.Tensor) error {
	return runAbsFloat16(args...)
}

func RunNegFloat16(args ...tensor.Tensor) error {
	return runNegFloat16(args...)
}

func RunSqrtFloat16(args ...tensor.Tensor) error {
	return runSqrtFloat16(args...)
}

func RunReluFloat16(args ...tensor.Tensor) error {
	return runReluFloat16(args...)
}
