//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

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
