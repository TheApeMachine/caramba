/*
Package cuda is the CUDA device backend. Host buffers stage through
cudaMallocHost (page-locked) for fast DMA; device storage uses
cudaMalloc. Upload is async on a dedicated upload stream; tensors
return from UploadAsync in StatePending and transition to StateReady
when the upload event fires.

Per the spray-and-pray contract (VERIFICATION_STATUS.md), this
package's shape matches the Backend contract in
pkg/backend/compute/tensor. The cgo bindings that actually call into
libcuda live in bridge_cuda.go (build-tagged) and return
ErrNeedsPlatformSetup elsewhere.
*/
package cuda

import (
	"sync/atomic"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Backend is the CUDA Backend implementation. The dtype set returned by
SupportedDTypes depends on the device's compute capability; H100 /
B200 add FP8E4M3 and FP8E5M2 to the base set.
*/
type Backend struct {
	closed atomic.Bool
	bridge *cudaBridge
}

/*
NewBackend constructs a CUDA backend. Returns ErrNeedsPlatformSetup
if no CUDA-capable device is present or the cgo toolchain is missing.
*/
func NewBackend() (*Backend, error) {
	bridge, err := openCUDABridge()

	if err != nil {
		return nil, err
	}

	return &Backend{bridge: bridge}, nil
}

/*
Location reports CUDA.
*/
func (backend *Backend) Location() tensor.Location {
	return tensor.CUDA
}

/*
SupportedDTypes lists the dtypes CUDA stores natively. Hopper /
Blackwell add Float8E4M3 and Float8E5M2 to the base list; the bridge
populates this from cudaGetDeviceProperties on construction.
*/
func (backend *Backend) SupportedDTypes() []dtype.DType {
	if backend.bridge == nil {
		return nil
	}

	return backend.bridge.supportedDTypes()
}

/*
SupportedLayouts: dense only until Phase 9 wires cuSPARSE.
*/
func (backend *Backend) SupportedLayouts() []tensor.Layout {
	return []tensor.Layout{tensor.LayoutDense}
}

/*
Capabilities returns the CUDA backend's properties. MaxBytes reflects
the device's totalGlobalMem; SupportsAsync is true; NativeAlignment is
128 bytes for coalesced access.
*/
func (backend *Backend) Capabilities() tensor.Capabilities {
	maxBytes := int64(0)

	if backend.bridge != nil {
		maxBytes = backend.bridge.totalGlobalMem()
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
Upload performs a synchronous H→D memcpy. Source dtypes outside
SupportedDTypes are rejected; callers convert through
pkg/dtype/convert first.
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
UploadAsync stages through a pinned host buffer and issues
cudaMemcpyAsync on a dedicated upload stream. Returned tensor is in
StatePending.
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
UploadSparse stubs until Phase 9 wires cuSPARSE.
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
Download issues a D→H memcpy through a pinned host buffer.
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
