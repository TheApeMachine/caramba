//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation

#include <stdlib.h>
#include <string.h>

// Forward declarations are intentionally minimal here; the full cgo
// binding lands in a follow-up session that can actually verify
// against Apple silicon. This file's role today is to give Phase 4
// a concrete shape: a real cgo handle that the future implementation
// hangs MTLBuffer creation, command queue allocation, and
// MTLResourceStorageModeShared upload off of.

typedef void* MetalDeviceRef;
typedef void* MetalBufferRef;

// Stub C functions; bodies live in bridge_darwin.m when this work
// lands fully. Phase 4 verification requires running on Apple silicon
// with the Metal command-line tools installed.
static MetalDeviceRef metal_open_default_device(void) { return NULL; }
static long long metal_recommended_max_working_set(MetalDeviceRef device) { return 0; }
static MetalBufferRef metal_buffer_new_shared(MetalDeviceRef device, long long bytes) { return NULL; }
static void metal_buffer_release(MetalBufferRef buffer) { (void)buffer; }
static void* metal_buffer_contents(MetalBufferRef buffer) { (void)buffer; return NULL; }
static void metal_device_release(MetalDeviceRef device) { (void)device; }
*/
import "C"

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
metalBridge wraps a MetalDeviceRef obtained from C.
metal_open_default_device. The Metal command queue is allocated lazily
on the first upload. Per spray-and-pray, this file compiles on
darwin+cgo but every operation returns ErrNeedsPlatformSetup until the
companion .m file lands; the surface is right, the bodies need to be
implemented and verified against real hardware.
*/
type metalBridge struct {
	device C.MetalDeviceRef
}

func openMetalBridge() (*metalBridge, error) {
	device := C.metal_open_default_device()

	if device == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	return &metalBridge{device: device}, nil
}

func (bridge *metalBridge) recommendedMaxWorkingSet() int64 {
	if bridge == nil || bridge.device == nil {
		return 0
	}

	return int64(C.metal_recommended_max_working_set(bridge.device))
}

func (bridge *metalBridge) upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *metalBridge) uploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *metalBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *metalBridge) close() error {
	if bridge.device != nil {
		C.metal_device_release(bridge.device)
		bridge.device = nil
	}

	return nil
}
