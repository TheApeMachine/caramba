/*
Package metal is the Metal device backend. On Apple silicon, host and
GPU share physical memory through MTLBuffer with
MTLResourceStorageModeShared, so Upload is metadata + memcpy with no
device-side narrowing.

Per the spray-and-pray contract (VERIFICATION_STATUS.md), this
package's shape matches the Backend contract in
pkg/backend/compute/tensor. The cgo bindings that actually call
MTLDevice and MTLBuffer live in metal_darwin.go (build-tagged) and
return ErrNeedsPlatformSetup elsewhere. The package compiles on every
platform; the body's behaviour depends on the build tag.
*/
package metal

import (
	"context"
	"sync/atomic"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Backend is the Metal Backend implementation. On darwin+cgo it routes
through real MTLDevice / MTLCommandQueue / MTLBuffer; otherwise it
returns ErrNeedsPlatformSetup for any operation that requires the
device.
*/
type Backend struct {
	closed atomic.Bool
	bridge *metalBridge
}

/*
NewBackend constructs a Metal backend. Returns ErrNeedsPlatformSetup
if the build is not darwin+cgo or the Metal device cannot be opened.
*/
func NewBackend() (*Backend, error) {
	bridge, err := openMetalBridge()

	if err != nil {
		return nil, err
	}

	return &Backend{bridge: bridge}, nil
}

/*
Location reports Metal.
*/
func (backend *Backend) Location() tensor.Location {
	return tensor.Metal
}

/*
SupportedDTypes lists the dtypes Metal stores natively (no conversion
at the boundary). Float32 / BFloat16 / Float16 are all GPU-resident.
Other dtypes are accepted via host-side conversion through
pkg/dtype/convert before upload.
*/
func (backend *Backend) SupportedDTypes() []dtype.DType {
	return []dtype.DType{
		dtype.Float32,
		dtype.BFloat16,
		dtype.Float16,
		dtype.Int8,
		dtype.Int4,
		dtype.Bool,
	}
}

/*
SupportedLayouts reports the layouts Metal can store directly. Phase 9
adds sparse paths through MPS-Graph sparse primitives; today the list
is dense-only.
*/
func (backend *Backend) SupportedLayouts() []tensor.Layout {
	return []tensor.Layout{tensor.LayoutDense}
}

/*
Capabilities reports the Metal backend's properties. MaxBytes is the
device's reported recommended working set; SupportsAsync is true when
the bridge is wired up; NativeAlignment is 256 bytes (Apple's
recommended buffer alignment for vector loads).
*/
func (backend *Backend) Capabilities() tensor.Capabilities {
	maxBytes := int64(0)

	if backend.bridge != nil {
		maxBytes = backend.bridge.recommendedMaxWorkingSet()
	}

	return tensor.Capabilities{
		MaxBytes:         maxBytes,
		SupportsAsync:    backend.bridge != nil,
		SupportsSparse:   false,
		SupportsAutograd: false,
		NativeAlignment:  256,
		NUMANodes:        1,
	}
}

/*
Upload moves bytes into an MTLBuffer with Shared storage mode and
returns a Tensor handle. When the source dtype is supported natively
the bytes are copied without conversion; otherwise convert.* is used
on the host side first. Per spray-and-pray, the bridge call returns
ErrNeedsPlatformSetup on non-Darwin builds.
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
UploadAsync hands the upload off to a dedicated Metal command queue
and returns a tensor in StatePending. The host caller observes
StateReady via Sync or Ready when the command buffer completes.
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
UploadSparse is a stub on the Metal backend until Phase 9.
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
Download materializes a host-side byte copy of the tensor's storage.
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
Close releases the bridge and marks the backend closed.
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

/*
SyncBlocking is a helper for tests that need to block on a tensor
becoming Ready. Production code should call Tensor.Sync directly.
*/
func SyncBlocking(ctx context.Context, target tensor.Tensor) error {
	return target.Sync(ctx)
}
