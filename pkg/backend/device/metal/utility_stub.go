//go:build !darwin || !cgo

package metal

import (
	"github.com/theapemachine/manifesto/tensor"
)

func runMetalCheckpointEncodeFloat32(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalCheckpointDecodeFloat32(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalTokenizerPackInt32(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalWeightFreezeMask(mask tensor.Tensor, gradients tensor.Tensor, out tensor.Tensor) error {
	_ = mask
	_ = gradients
	_ = out

	return tensor.ErrNeedsPlatformSetup
}
