/*
Package xla is the XLA device backend. Tensors are HLO buffers managed
through the XLA runtime's transfer manager. Upload is async by default;
Download blocks until the transfer event fires.

Per the spray-and-pray contract, this package compiles on every
platform; the XLA runtime bindings live in bridge_xla.go behind the
'xla' build tag and return ErrNeedsPlatformSetup elsewhere.
*/
package xla

import (
	"sync/atomic"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Backend is the XLA Backend implementation.
*/
type Backend struct {
	closed atomic.Bool
	bridge *xlaBridge
}

/*
NewBackend constructs an XLA backend. Returns ErrNeedsPlatformSetup
if the XLA runtime cannot be opened on the current platform.
*/
func NewBackend() (*Backend, error) {
	bridge, err := openXLABridge()

	if err != nil {
		return nil, err
	}

	return &Backend{bridge: bridge}, nil
}

/*
Location reports XLA.
*/
func (backend *Backend) Location() tensor.Location {
	return tensor.XLA
}

/*
SupportedDTypes returns the XLA-native dtype set, which is the
broadest of the device backends: every IEEE 754 float plus the
quantized integer family.
*/
func (backend *Backend) SupportedDTypes() []dtype.DType {
	return []dtype.DType{
		dtype.Float64,
		dtype.Float32,
		dtype.Float16,
		dtype.BFloat16,
		dtype.Float8E4M3,
		dtype.Float8E5M2,
		dtype.Int64,
		dtype.Int32,
		dtype.Int16,
		dtype.Int8,
		dtype.Uint64,
		dtype.Uint32,
		dtype.Uint16,
		dtype.Uint8,
		dtype.Bool,
	}
}

/*
SupportedLayouts: dense only until Phase 9.
*/
func (backend *Backend) SupportedLayouts() []tensor.Layout {
	return []tensor.Layout{tensor.LayoutDense}
}

/*
Capabilities returns the XLA backend's properties. MaxBytes depends on
the platform (TPU pod vs. XLA-GPU); the bridge fills it in on open.
*/
func (backend *Backend) Capabilities() tensor.Capabilities {
	maxBytes := int64(0)

	if backend.bridge != nil {
		maxBytes = backend.bridge.devicePoolBytes()
	}

	return tensor.Capabilities{
		MaxBytes:         maxBytes,
		SupportsAsync:    backend.bridge != nil,
		SupportsSparse:   false,
		SupportsAutograd: false,
		NativeAlignment:  128,
		NUMANodes:        1,
	}
}

/*
Upload routes through the XLA transfer manager. Synchronous variant.
*/
func (backend *Backend) Upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	if backend.closed.Load() {
		return nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.upload(shape, sourceDType, bytesIn)
}

/*
UploadAsync issues an XLA async transfer. The returned tensor is in
StatePending until the transfer event fires.
*/
func (backend *Backend) UploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	if backend.closed.Load() {
		return nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.uploadAsync(shape, sourceDType, bytesIn)
}

/*
UploadSparse stubs until Phase 9.
*/
func (backend *Backend) UploadSparse(
	shape tensor.Shape,
	valueDType dtype.DType,
	layout tensor.Layout,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}

/*
Download materializes a host copy of the tensor's storage.
*/
func (backend *Backend) Download(input tensor.Tensor) (dtype.DType, []byte, error) {
	if backend.closed.Load() {
		return dtype.Invalid, nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.download(input)
}

/*
Close releases the bridge.
*/
func (backend *Backend) Close() error {
	if !backend.closed.CompareAndSwap(false, true) {
		return nil
	}

	if backend.bridge != nil {
		err := backend.bridge.close()
		backend.bridge = nil
		return err
	}

	return nil
}

var _ tensor.Backend = (*Backend)(nil)
