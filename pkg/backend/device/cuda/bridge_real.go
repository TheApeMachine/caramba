//go:build cuda

package cuda

/*
#cgo LDFLAGS: -lcuda -lcudart

#include <stdlib.h>
#include <string.h>

// Forward declarations for the CUDA runtime bindings. The full
// implementation requires linking against libcuda and libcudart and
// running on a CUDA-capable host. Phase 5 verification needs an H100
// or B200 (for FP8 paths) or any sm_70+ device for the base set.

typedef void* CUDADeviceRef;
typedef void* CUDABufferRef;
typedef void* CUDAStreamRef;

static CUDADeviceRef cuda_open_default_device(void) { return NULL; }
static long long cuda_total_global_mem(CUDADeviceRef device) { (void)device; return 0; }
static int cuda_compute_capability_major(CUDADeviceRef device) { (void)device; return 0; }
static int cuda_compute_capability_minor(CUDADeviceRef device) { (void)device; return 0; }
static CUDABufferRef cuda_buffer_alloc(CUDADeviceRef device, long long bytes) { (void)device; (void)bytes; return NULL; }
static void cuda_buffer_release(CUDABufferRef buffer) { (void)buffer; }
static CUDAStreamRef cuda_stream_create(CUDADeviceRef device) { (void)device; return NULL; }
static void cuda_stream_destroy(CUDAStreamRef stream) { (void)stream; }
static int cuda_memcpy_async_h2d(CUDABufferRef dst, void* src, long long bytes, CUDAStreamRef stream) {
    (void)dst; (void)src; (void)bytes; (void)stream;
    return 0;
}
*/
import "C"

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
cudaBridge wraps a CUDA device handle plus the upload stream. The
supported-dtype list depends on the device's compute capability; the
bridge constructs it once on open.
*/
type cudaBridge struct {
	device       C.CUDADeviceRef
	uploadStream C.CUDAStreamRef
	dtypes       []dtype.DType
	totalMem     int64
}

func openCUDABridge() (*cudaBridge, error) {
	device := C.cuda_open_default_device()

	if device == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	stream := C.cuda_stream_create(device)

	if stream == nil {
		C.cuda_buffer_release(C.CUDABufferRef(device))
		return nil, tensor.ErrNeedsPlatformSetup
	}

	major := int(C.cuda_compute_capability_major(device))
	totalMem := int64(C.cuda_total_global_mem(device))

	supported := []dtype.DType{
		dtype.Float32,
		dtype.BFloat16,
		dtype.Float16,
		dtype.Int8,
		dtype.Int4,
		dtype.Bool,
	}

	if major >= 9 {
		// Hopper / Blackwell add native FP8 paths.
		supported = append(supported, dtype.Float8E4M3, dtype.Float8E5M2)
	}

	return &cudaBridge{
		device:       device,
		uploadStream: stream,
		dtypes:       supported,
		totalMem:     totalMem,
	}, nil
}

func (bridge *cudaBridge) supportedDTypes() []dtype.DType {
	return bridge.dtypes
}

func (bridge *cudaBridge) totalGlobalMem() int64 {
	return bridge.totalMem
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
	if bridge.uploadStream != nil {
		C.cuda_stream_destroy(bridge.uploadStream)
		bridge.uploadStream = nil
	}

	bridge.device = nil
	return nil
}
