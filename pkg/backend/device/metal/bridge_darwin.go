//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include <stdlib.h>
#include <string.h>
#include "bridge_darwin.h"
*/
import "C"

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

//go:embed kernels.metallib
var kernelsMetalLibrary []byte

/*
metalBridge wraps a Metal device, command queue, and compiled kernel
pipelines.
*/
type metalBridge struct {
	device C.MetalDeviceRef
}

func openMetalBridge() (*metalBridge, error) {
	if len(kernelsMetalLibrary) == 0 {
		return nil, fmt.Errorf("%w: empty Metal library", tensor.ErrNeedsPlatformSetup)
	}

	libraryBytes := C.CBytes(kernelsMetalLibrary)
	if libraryBytes == nil {
		return nil, fmt.Errorf("%w: Metal library allocation failed", tensor.ErrNeedsPlatformSetup)
	}
	defer C.free(libraryBytes)

	status := C.MetalStatus{}
	device := C.metal_open_default_device(
		(*C.uint8_t)(libraryBytes),
		C.longlong(len(kernelsMetalLibrary)),
		&status,
	)

	if device == nil {
		return nil, fmt.Errorf("%w: %s", tensor.ErrNeedsPlatformSetup, metalStatus("open", status))
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
	expectedBytes, err := shape.Bytes(sourceDType)
	if err != nil {
		return nil, err
	}

	if expectedBytes != len(bytesIn) {
		return nil, tensor.ErrShapeMismatch
	}

	target, err := bridge.empty(shape, sourceDType)
	if err != nil {
		return nil, err
	}

	if len(bytesIn) == 0 {
		return target, nil
	}

	contents := C.metal_buffer_contents(target.buffer)
	if contents == nil {
		_ = target.Close()
		return nil, tensor.ErrNeedsPlatformSetup
	}

	C.memcpy(contents, unsafe.Pointer(&bytesIn[0]), C.size_t(len(bytesIn)))

	return target, nil
}

func (bridge *metalBridge) uploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return bridge.upload(shape, sourceDType, bytesIn)
}

func (bridge *metalBridge) empty(
	shape tensor.Shape,
	storageDType dtype.DType,
) (*metalTensor, error) {
	bytes, err := shape.Bytes(storageDType)
	if err != nil {
		return nil, err
	}

	var buffer C.MetalBufferRef

	if bytes > 0 {
		buffer = C.metal_buffer_new_shared(bridge.device, C.longlong(bytes))
		if buffer == nil {
			return nil, tensor.ErrAllocatorExhausted
		}
	}

	ready := make(chan struct{})
	close(ready)

	return &metalTensor{
		bridge: bridge,
		shape:  shape,
		dtype:  storageDType,
		buffer: buffer,
		bytes:  bytes,
		ready:  ready,
	}, nil
}

func (bridge *metalBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	target, err := requireMetalTensor(input)
	if err != nil {
		return dtype.Invalid, nil, err
	}

	out := make([]byte, target.bytes)
	if len(out) == 0 {
		return target.dtype, out, nil
	}

	contents := C.metal_buffer_contents(target.buffer)
	if contents == nil {
		return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
	}

	copy(out, unsafe.Slice((*byte)(contents), len(out)))

	return target.dtype, out, nil
}

func (bridge *metalBridge) close() error {
	if bridge.device != nil {
		C.metal_device_release(bridge.device)
		bridge.device = nil
	}

	return nil
}

func runMetalAddFloat32(left, right, out tensor.Tensor) error {
	leftTensor, err := requireMetalTensor(left)
	if err != nil {
		return err
	}

	rightTensor, err := requireMetalTensor(right)
	if err != nil {
		return err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return err
	}

	if leftTensor.dtype != dtype.Float32 ||
		rightTensor.dtype != dtype.Float32 ||
		outTensor.dtype != dtype.Float32 {
		return tensor.ErrDTypeMismatch
	}

	if leftTensor.shape.Len() != rightTensor.shape.Len() ||
		leftTensor.shape.Len() != outTensor.shape.Len() {
		return tensor.ErrShapeMismatch
	}

	if !leftTensor.shape.Equal(rightTensor.shape) || !leftTensor.shape.Equal(outTensor.shape) {
		return tensor.ErrShapeMismatch
	}

	if leftTensor.shape.Len() > math.MaxUint32 {
		return tensor.ErrShapeMismatch
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_add_float32(
		leftTensor.bridge.device,
		leftTensor.buffer,
		rightTensor.buffer,
		outTensor.buffer,
		C.uint32_t(leftTensor.shape.Len()),
		&status,
	)

	if rc != 0 {
		return fmt.Errorf("metal add float32: %s", metalStatus("dispatch", status))
	}

	return nil
}

func metalStatus(operation string, status C.MetalStatus) string {
	message := C.GoString(&status.message[0])
	if message == "" {
		message = "unknown error"
	}

	return fmt.Sprintf("%s failed: %s (code=%d)", operation, message, int(status.code))
}

func requireMetalTensor(input tensor.Tensor) (*metalTensor, error) {
	if input == nil {
		return nil, errors.New("metal tensor: nil input")
	}

	target, ok := input.(*metalTensor)
	if !ok {
		return nil, fmt.Errorf("metal tensor: expected metal tensor, got %T", input)
	}

	if target.closed.Load() {
		return nil, tensor.ErrTensorClosed
	}

	return target, nil
}

type metalTensor struct {
	bridge *metalBridge
	shape  tensor.Shape
	dtype  dtype.DType
	buffer C.MetalBufferRef
	bytes  int
	closed atomic.Bool
	ready  chan struct{}
}

func (target *metalTensor) Shape() tensor.Shape {
	return target.shape
}

func (target *metalTensor) DType() dtype.DType {
	return target.dtype
}

func (target *metalTensor) Layout() tensor.Layout {
	return tensor.LayoutDense
}

func (target *metalTensor) Location() tensor.Location {
	return tensor.Metal
}

func (target *metalTensor) Len() int {
	return target.shape.Len()
}

func (target *metalTensor) Bytes() int {
	return target.bytes
}

func (target *metalTensor) Close() error {
	if !target.closed.CompareAndSwap(false, true) {
		return nil
	}

	if target.buffer != nil {
		C.metal_buffer_release(target.buffer)
		target.buffer = nil
	}

	return nil
}

func (target *metalTensor) Slice(start, length int) (tensor.Tensor, error) {
	_ = start
	_ = length

	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Reshape(dims []int) (tensor.Tensor, error) {
	_ = dims

	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Float64Native() ([]float64, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Float32Native() ([]float32, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Float16Native() ([]dtype.F16, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) BFloat16Native() ([]dtype.BF16, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Float8E4M3Native() ([]dtype.F8E4M3, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Float8E5M2Native() ([]dtype.F8E5M2, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Int64Native() ([]int64, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Int32Native() ([]int32, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Int16Native() ([]int16, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Int8Native() ([]int8, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Uint64Native() ([]uint64, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Uint32Native() ([]uint32, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Uint16Native() ([]uint16, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Uint8Native() ([]uint8, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) BoolNative() (tensor.BitVector, error) {
	return tensor.BitVector{}, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) Int4Native() (tensor.Int4Vector, error) {
	return tensor.Int4Vector{}, tensor.ErrLayoutUnsupported
}

func (target *metalTensor) RawBytes() (dtype.DType, []byte, error) {
	if target.bridge == nil {
		return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
	}

	return target.bridge.download(target)
}

func (target *metalTensor) State() tensor.State {
	if target.closed.Load() {
		return tensor.StateClosed
	}

	return tensor.StateReady
}

func (target *metalTensor) Sync(ctx context.Context) error {
	return ctx.Err()
}

func (target *metalTensor) Ready() <-chan struct{} {
	return target.ready
}

func (target *metalTensor) RequiresGrad() bool {
	return false
}

func (target *metalTensor) SetRequiresGrad(yes bool) error {
	_ = yes

	return tensor.ErrBackwardNotImplemented
}

func (target *metalTensor) Grad() (tensor.Tensor, error) {
	return nil, tensor.ErrNoAutograd
}

func (target *metalTensor) GradFn() tensor.GradFn {
	return nil
}
