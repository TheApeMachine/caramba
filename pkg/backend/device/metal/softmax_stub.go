//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/manifesto/tensor"

func runMetalSoftmax(input tensor.Tensor, out tensor.Tensor) error {
	return tensor.ErrNeedsPlatformSetup
}
