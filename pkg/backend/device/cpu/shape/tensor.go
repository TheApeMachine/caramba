package shape

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func RunWhereFloat32(args ...tensor.Tensor) error {
	return runWhereFloat32(args...)
}

func RunMaskedFillFloat32(args ...tensor.Tensor) error {
	return runMaskedFillFloat32(args...)
}

func RunTranspose(args ...tensor.Tensor) error {
	return runTranspose(args...)
}

func RunReshape(args ...tensor.Tensor) error {
	return runReshape(args...)
}

func RunUpsampleNearest2D(args ...tensor.Tensor) error {
	return runUpsampleNearest2D(args...)
}

func RunLastToken(args ...tensor.Tensor) error {
	return runLastToken(args...)
}

func RunMergeHeads(args ...tensor.Tensor) error {
	return runMergeHeads(args...)
}

func RunSplitHeads(args ...tensor.Tensor) error {
	return runSplitHeads(args...)
}

func RunSplit2(args ...tensor.Tensor) error {
	return runSplit2(args...)
}

func RunViewAsHeads(args ...tensor.Tensor) error {
	return runViewAsHeads(args...)
}
