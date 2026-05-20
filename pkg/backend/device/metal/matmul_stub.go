//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/manifesto/tensor"

func runMetalMatMul(left tensor.Tensor, right tensor.Tensor, out tensor.Tensor) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalMatMulAdd(
	left tensor.Tensor,
	right tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}
