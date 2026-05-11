//go:build darwin && cgo

package metal

// #include "activation.h"
// #include "tensor.h"
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ computetensor.Backend = (*TensorBackend)(nil)

/*
TensorBackend owns Metal MTLBuffer tensors.
*/
type TensorBackend struct {
	closed bool
}

/*
NewTensorBackend creates a Metal resident tensor backend.
*/
func NewTensorBackend() (*TensorBackend, error) {
	if rc := C.metal_tensor_init(); rc != 0 {
		return nil, fmt.Errorf("metal tensor: initialization failed")
	}

	return &TensorBackend{}, nil
}

/*
Location identifies Metal storage ownership.
*/
func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.Metal
}

/*
UploadFloat64 converts host float64 values into resident Metal float32 storage.
*/
func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	if tensorBackend.closed {
		return nil, errors.New("metal tensor: backend is closed")
	}

	if !shape.Valid() {
		return nil, errors.New("metal tensor: invalid shape")
	}

	if shape.Len() != len(values) {
		return nil, fmt.Errorf(
			"metal tensor: shape has %d elements but upload received %d values",
			shape.Len(), len(values),
		)
	}

	float32Values := toFloat32(values)
	bytes, err := shape.Bytes(computetensor.Float32)

	if err != nil {
		return nil, err
	}

	var buffer unsafe.Pointer

	if len(float32Values) > 0 {
		buffer = C.metal_tensor_upload_float32(
			(*C.float)(unsafe.Pointer(&float32Values[0])),
			C.int(len(float32Values)),
		)

		if buffer == nil {
			return nil, fmt.Errorf("metal tensor: upload failed")
		}
	}

	return &Tensor{
		bytes:  bytes,
		shape:  shape,
		buffer: buffer,
	}, nil
}

/*
DownloadFloat64 copies resident Metal storage back to host float64 values.
*/
func (tensorBackend *TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	if tensorBackend.closed {
		return nil, errors.New("metal tensor: backend is closed")
	}

	if input == nil {
		return nil, errors.New("metal tensor: nil input")
	}

	if input.Location() != computetensor.Metal {
		return nil, fmt.Errorf("metal tensor: cannot download %s tensor", input.Location())
	}

	return input.CloneFloat64()
}

/*
Close releases the backend.
*/
func (tensorBackend *TensorBackend) Close() error {
	tensorBackend.closed = true

	return nil
}

/*
ReLUTensor launches a Metal ReLU kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) ReLUTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "relu", 0)
}

/*
LeakyReLUTensor launches a Metal leaky ReLU kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) LeakyReLUTensor(
	input computetensor.Float64Tensor, alpha float64,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "leaky_relu", alpha)
}

/*
GELUTensor launches a Metal GELU kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) GELUTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "gelu", 0)
}

/*
TanhTensor launches a Metal tanh kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) TanhTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "tanh", 0)
}

/*
SigmoidTensor launches a Metal sigmoid kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) SigmoidTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "sigmoid", 0)
}

/*
SwiGLUTensor launches a Metal SwiGLU kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) SwiGLUTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	outputShape, err := metalSwiGLUOutputShape(metalInput.shape)

	if err != nil {
		return nil, err
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_swiglu_tensor(metalInput.buffer, output.buffer, C.int(outputShape.Len()))

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: swiglu launch failed")
	}

	return output, nil
}

func (metalActivation *MetalActivation) unaryTensor(
	input computetensor.Float64Tensor, name string, alpha float64,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	output, err := newMetalTensor(metalInput.shape)

	if err != nil {
		return nil, err
	}

	var rc C.int

	switch name {
	case "relu":
		rc = C.metal_relu_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	case "leaky_relu":
		rc = C.metal_leaky_relu_tensor(
			metalInput.buffer, output.buffer, C.float(alpha), C.int(metalInput.Len()),
		)
	case "gelu":
		rc = C.metal_gelu_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	case "tanh":
		rc = C.metal_tanh_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	case "sigmoid":
		rc = C.metal_sigmoid_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	default:
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: unknown unary kernel %q", name)
	}

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: %s launch failed", name)
	}

	return output, nil
}

func newMetalTensor(shape computetensor.Shape) (*Tensor, error) {
	bytes, err := shape.Bytes(computetensor.Float32)

	if err != nil {
		return nil, err
	}

	var buffer unsafe.Pointer

	if shape.Len() > 0 {
		buffer = C.metal_tensor_empty_float32(C.int(shape.Len()))

		if buffer == nil {
			return nil, fmt.Errorf("metal tensor: allocation of %d bytes failed", bytes)
		}
	}

	return &Tensor{
		bytes:  bytes,
		shape:  shape,
		buffer: buffer,
	}, nil
}

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

	if metalInput.closed {
		return nil, errors.New("metal tensor: input is closed")
	}

	return metalInput, nil
}

/*
Tensor is persistent Metal MTLBuffer storage.
*/
type Tensor struct {
	bytes  int
	shape  computetensor.Shape
	buffer unsafe.Pointer
	closed bool
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
	return computetensor.Float32
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
CloneFloat64 downloads Metal float32 storage into host float64 values.
*/
func (tensor *Tensor) CloneFloat64() ([]float64, error) {
	if tensor.closed {
		return nil, errors.New("metal tensor: tensor is closed")
	}

	values := make([]float32, tensor.Len())

	if len(values) == 0 {
		return []float64{}, nil
	}

	rc := C.metal_tensor_download_float32(
		tensor.buffer,
		(*C.float)(unsafe.Pointer(&values[0])),
		C.int(len(values)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal tensor: download failed")
	}

	return toFloat64(values), nil
}

/*
Close releases the MTLBuffer.
*/
func (tensor *Tensor) Close() error {
	if tensor.closed {
		return nil
	}

	rc := C.metal_tensor_free(tensor.buffer)
	tensor.bytes = 0
	tensor.buffer = nil
	tensor.closed = true

	if rc != 0 {
		return fmt.Errorf("metal tensor: free failed")
	}

	return nil
}

func metalSwiGLUOutputShape(shape computetensor.Shape) (computetensor.Shape, error) {
	dims := shape.Dims()

	if shape.Len()%2 != 0 {
		return computetensor.Shape{}, fmt.Errorf("metal tensor: swiglu input length must be even")
	}

	if len(dims) == 0 {
		return computetensor.NewShape([]int{shape.Len() / 2})
	}

	lastIndex := len(dims) - 1

	if dims[lastIndex]%2 != 0 {
		return computetensor.Shape{}, fmt.Errorf("metal tensor: swiglu final dimension must be even")
	}

	dims[lastIndex] /= 2

	return computetensor.NewShape(dims)
}
