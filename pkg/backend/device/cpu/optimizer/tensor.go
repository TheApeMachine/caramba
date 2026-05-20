package optimizer

import "github.com/theapemachine/manifesto/tensor"

func RunAdamStepDefault(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return AdamStepFloat32(
		DefaultAdamConfig(),
		args[0], args[1], args[2], args[3], args[4],
	)
}

func RunAdamWStepDefault(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return AdamWStepFloat32(
		DefaultAdamWConfig(),
		args[0], args[1], args[2], args[3], args[4],
	)
}

func RunLionStepDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return LionStepFloat32(
		DefaultLionConfig(),
		args[0], args[1], args[2], args[3],
	)
}

func RunSGDStepDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return SGDStepFloat32(
		DefaultSGDConfig(),
		args[0], args[1], args[2], args[3],
	)
}

func RunAdamaxStepDefault(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return AdamaxStepFloat32(
		DefaultAdamaxConfig(),
		args[0], args[1], args[2], args[3], args[4],
	)
}

func RunAdagradStepDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return AdagradStepFloat32(
		DefaultAdagradConfig(),
		args[0], args[1], args[2], args[3],
	)
}

func RunRMSpropStepDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return RMSpropStepFloat32(
		DefaultRMSpropConfig(),
		args[0], args[1], args[2], args[3],
	)
}

func RunLARSStepDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return LARSStepFloat32(DefaultLARSConfig(), args[0], args[1], args[2], args[3])
}

func RunHebbianStepDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return HebbianStepFloat32(DefaultHebbianConfig(), args[0], args[1], args[2], args[3])
}

func RunLBFGSStepDefault(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return LBFGSStepFloat32(DefaultLBFGSConfig(), args[0], args[1], args[2])
}
