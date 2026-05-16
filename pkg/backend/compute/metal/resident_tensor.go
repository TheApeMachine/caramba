//go:build darwin && cgo

package metal

// #include "tensor.h"
import "C"

import (
	"errors"
	"fmt"
	"sync/atomic"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func requireMetalTensor(input computetensor.Float64Tensor) (*Tensor, error) {
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
	bytes    int
	shape    computetensor.Shape
	buffer   unsafe.Pointer
	runtime  *MetalRuntime
	metadata MetalTensorMetadata
	closed   atomic.Uint32
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
func (tensor *Tensor) DType() computetensor.DType {
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
func (tensor *Tensor) Layout() MetalLayout {
	return tensor.metadata.Layout
}

/*
CloneFloat64 downloads Metal float32 storage into host float64 values.
*/
func (tensor *Tensor) CloneFloat64() ([]float64, error) {
	if tensor.closed.Load() != 0 {
		return nil, errors.New("metal tensor: tensor is closed")
	}

	values := make([]float32, tensor.Len())

	if len(values) == 0 {
		return []float64{}, nil
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

	return toFloat64(values), nil
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
