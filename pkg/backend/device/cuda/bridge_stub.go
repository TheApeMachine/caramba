//go:build !cuda

package cuda

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
cudaBridge stub for builds without the 'cuda' tag. Every method
returns ErrNeedsPlatformSetup so callers compile but the device is
clearly unavailable. The real bridge with cgo bindings to libcuda
lives in bridge_real.go behind //go:build cuda.
*/
type cudaBridge struct{}

func openCUDABridge() (*cudaBridge, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *cudaBridge) supportedDTypes() []dtype.DType {
	return nil
}

func (bridge *cudaBridge) totalGlobalMem() int64 {
	return 0
}

func (bridge *cudaBridge) upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *cudaBridge) uploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *cudaBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *cudaBridge) close() error {
	return nil
}
