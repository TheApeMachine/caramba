//go:build darwin && cgo

package metal

// #include "tensor.h"
import "C"

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func requireMetalTensor(input computetensor.Tensor) (*Tensor, error) {
	if input == nil {
		return nil, errors.New("metal tensor: nil input")
	}

	if input.Location() != computetensor.Metal {
		return nil, fmt.Errorf("metal tensor: cannot execute %s tensor", input.Location())
	}

	metalInput, ok := input.(*Tensor)

	if !ok {
		return nil, fmt.Errorf("metal tensor: input is not owned by Metal backend")
	}

	if metalInput.closed.Load() != 0 {
		return nil, errors.New("metal tensor: input is closed")
	}

	return metalInput, nil
}

/*
Tensor is persistent Metal MTLBuffer storage.
*/
type Tensor struct {
	bytes     int
	shape     computetensor.Shape
	buffer    unsafe.Pointer
	runtime   *MetalRuntime
	metadata  MetalTensorMetadata
	accounted bool
	closed    atomic.Uint32
}

/*
Shape returns validated tensor dimensions.
*/
func (tensor *Tensor) Shape() computetensor.Shape {
	return tensor.shape
}

/*
DType reports float32 Metal storage.
*/
func (tensor *Tensor) DType() dtype.DType {
	return tensor.metadata.DType
}

/*
Location reports Metal ownership.
*/
func (tensor *Tensor) Location() computetensor.Location {
	return computetensor.Metal
}

/*
Len reports the number of tensor elements.
*/
func (tensor *Tensor) Len() int {
	return tensor.shape.Len()
}

/*
Bytes reports the Metal buffer allocation size.
*/
func (tensor *Tensor) Bytes() int {
	return tensor.bytes
}

/*
Metadata returns dtype, layout, and storage policy for the resident tensor.
*/
func (tensor *Tensor) Metadata() MetalTensorMetadata {
	return cloneMetadata(tensor.metadata)
}

/*
StorageMode reports the Metal storage mode for this tensor.
*/
func (tensor *Tensor) StorageMode() MetalStorageMode {
	return tensor.metadata.StorageMode
}

/*
Strides returns the physical contiguous strides recorded for this tensor.
*/
func (tensor *Tensor) Strides() []int {
	return append([]int(nil), tensor.metadata.Strides...)
}

/*
Layout reports the physical layout family.
*/
func (tensor *Tensor) Layout() computetensor.Layout {
	return computetensor.LayoutDense
}

/*
MetalLayout reports the backend-specific Metal layout metadata.
*/
func (tensor *Tensor) MetalLayout() MetalLayout {
	return tensor.metadata.Layout
}

/*
downloadFloat32 copies resident float32 storage to host.
*/
func (tensor *Tensor) downloadFloat32() ([]float32, error) {
	if tensor.closed.Load() != 0 {
		return nil, errors.New("metal tensor: tensor is closed")
	}

	values := make([]float32, tensor.Len())

	if len(values) == 0 {
		return []float32{}, nil
	}

	rc := C.metal_tensor_download_float32(
		tensor.buffer,
		(*C.float)(unsafe.Pointer(&values[0])),
		C.size_t(len(values)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal tensor: download failed (rc=%d)", rc)
	}

	tensor.runtime.recordTransfer(tensor.bytes)

	return values, nil
}

func (tensor *Tensor) Slice(start, length int) (computetensor.Tensor, error) {
	_ = start
	_ = length

	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Reshape(dims []int) (computetensor.Tensor, error) {
	_ = dims

	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Float64Native() ([]float64, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Float32Native() ([]float32, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Float16Native() ([]dtype.F16, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) BFloat16Native() ([]dtype.BF16, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Float8E4M3Native() ([]dtype.F8E4M3, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Float8E5M2Native() ([]dtype.F8E5M2, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Int64Native() ([]int64, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Int32Native() ([]int32, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Int16Native() ([]int16, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Int8Native() ([]int8, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Uint64Native() ([]uint64, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Uint32Native() ([]uint32, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Uint16Native() ([]uint16, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Uint8Native() ([]uint8, error) {
	return nil, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) BoolNative() (computetensor.BitVector, error) {
	return computetensor.BitVector{}, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) Int4Native() (computetensor.Int4Vector, error) {
	return computetensor.Int4Vector{}, computetensor.ErrLayoutUnsupported
}

func (tensor *Tensor) RawBytes() (dtype.DType, []byte, error) {
	values, err := tensor.downloadFloat32()
	if err != nil {
		return dtype.Invalid, nil, err
	}

	return dtype.Float32, dtypeconvert.Float32ToBytes(values), nil
}

func (tensor *Tensor) State() computetensor.State {
	if tensor.closed.Load() != 0 {
		return computetensor.StateClosed
	}

	return computetensor.StateReady
}

func (tensor *Tensor) Sync(ctx context.Context) error {
	return ctx.Err()
}

func (tensor *Tensor) Ready() <-chan struct{} {
	ready := make(chan struct{})
	close(ready)

	return ready
}

func (tensor *Tensor) RequiresGrad() bool {
	return false
}

func (tensor *Tensor) SetRequiresGrad(yes bool) error {
	_ = yes

	return computetensor.ErrBackwardNotImplemented
}

func (tensor *Tensor) Grad() (computetensor.Tensor, error) {
	return nil, computetensor.ErrNoAutograd
}

func (tensor *Tensor) GradFn() computetensor.GradFn {
	return nil
}

/*
Close releases the MTLBuffer.
*/
func (tensor *Tensor) Close() error {
	if !tensor.closed.CompareAndSwap(0, 1) {
		return nil
	}

	bufferPtr := tensor.buffer

	if bufferPtr == nil {
		return nil
	}

	return tensor.runtime.release(tensor)
}
