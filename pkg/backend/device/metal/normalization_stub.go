//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/manifesto/tensor"

func runMetalLayerNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalRMSNorm(input tensor.Tensor, scale tensor.Tensor, out tensor.Tensor) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalGroupNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalInstanceNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalBatchNormEval(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	mean tensor.Tensor,
	variance tensor.Tensor,
	out tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}
