//go:build !xla

package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
xlaBridge stub for builds without the 'xla' tag.
*/
type xlaBridge struct{}

func openXLABridge() (*xlaBridge, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *xlaBridge) devicePoolBytes() int64 {
	return 0
}

func (bridge *xlaBridge) upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *xlaBridge) uploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *xlaBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *xlaBridge) close() error {
	return nil
}
