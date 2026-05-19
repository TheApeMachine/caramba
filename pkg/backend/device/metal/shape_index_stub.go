//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func runMetalGather(source tensor.Tensor, indices tensor.Tensor, out tensor.Tensor) error {
	_ = source
	_ = indices
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalScatter(
	target tensor.Tensor,
	indices tensor.Tensor,
	updates tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = target
	_ = indices
	_ = updates
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalWhere(
	mask tensor.Tensor,
	positive tensor.Tensor,
	negative tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = mask
	_ = positive
	_ = negative
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalMaskedFill(
	input tensor.Tensor,
	mask tensor.Tensor,
	scalar tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = input
	_ = mask
	_ = scalar
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalTranspose(input tensor.Tensor, permutation tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = permutation
	_ = out

	return tensor.ErrNeedsPlatformSetup
}
