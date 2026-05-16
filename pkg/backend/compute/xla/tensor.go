//go:build cgo && xla

package xla

// #include <stdint.h>
// #include <stdlib.h>
// #include "tensor.h"
import "C"

import (
	"errors"
	"fmt"
	"slices"
	"sync/atomic"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ computetensor.Backend = (*TensorBackend)(nil)

/*
TensorBackend owns PJRT buffers for resident XLA execution.
*/
type TensorBackend struct {
	platform string
	closed   atomic.Uint32
}

/*
NewTensorBackend creates an XLA/PJRT resident tensor backend.
*/
func NewTensorBackend(platform string) (*TensorBackend, error) {
	config, err := newRuntimePJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	platformCString := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(platformCString))

	if rc := C.xla_tensor_init(platformCString); rc != 0 {
		return nil, fmt.Errorf("xla tensor: initialization failed for platform %q", config.Platform)
	}

	return &TensorBackend{platform: config.Platform}, nil
}

/*
Location identifies XLA PJRT storage ownership.
*/
func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.XLA
}

/*
UploadFloat64 copies host values into a resident PJRT buffer.
*/
func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	if tensorBackend.closed.Load() != 0 {
		return nil, errors.New("xla tensor: backend is closed")
	}

	if !shape.Valid() {
		return nil, errors.New("xla tensor: invalid shape")
	}

	if shape.Len() != len(values) {
		return nil, fmt.Errorf(
			"xla tensor: shape has %d elements but upload received %d values",
			shape.Len(), len(values),
		)
	}

	bytes, err := shape.Bytes(computetensor.Float64)

	if err != nil {
		return nil, err
	}

	cDims := cShapeDims(shape)
	var cDimsPointer *C.int64_t

	if len(cDims) > 0 {
		cDimsPointer = &cDims[0]
	}

	var valuesPointer *C.double

	if len(values) > 0 {
		valuesPointer = (*C.double)(unsafe.Pointer(&values[0]))
	}

	var handle *C.XLA_Tensor

	rc := C.xla_tensor_upload_f64(
		valuesPointer,
		cDimsPointer,
		C.int(len(cDims)),
		(**C.XLA_Tensor)(unsafe.Pointer(&handle)),
	)

	if rc != 0 || handle == nil {
		return nil, fmt.Errorf("xla tensor: upload failed")
	}

	return &Tensor{
		bytes:  bytes,
		shape:  shape,
		handle: handle,
	}, nil
}

/*
DownloadFloat64 copies a resident PJRT buffer back into host memory.
*/
func (tensorBackend *TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	if tensorBackend.closed.Load() != 0 {
		return nil, errors.New("xla tensor: backend is closed")
	}

	xlaInput, err := tensorBackend.require(input)

	if err != nil {
		return nil, err
	}

	return xlaInput.CloneFloat64()
}

/*
Close releases the backend executable cache.
*/
func (tensorBackend *TensorBackend) Close() error {
	if !tensorBackend.closed.CompareAndSwap(0, 1) {
		return nil
	}

	C.xla_tensor_shutdown()

	return nil
}

/*
ReLU executes max(0, x) against a resident PJRT buffer.
*/
func (tensorBackend *TensorBackend) ReLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "relu", 0)
}

/*
LeakyReLU executes max(alpha*x, x) against a resident PJRT buffer.
*/
func (tensorBackend *TensorBackend) LeakyReLU(
	input computetensor.Float64Tensor, alpha float64,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "leaky_relu", alpha)
}

/*
GELU executes tanh-approximate GELU against a resident PJRT buffer.
*/
func (tensorBackend *TensorBackend) GELU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "gelu", 0)
}

/*
Tanh executes hyperbolic tangent against a resident PJRT buffer.
*/
func (tensorBackend *TensorBackend) Tanh(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "tanh", 0)
}

/*
Sigmoid executes logistic sigmoid against a resident PJRT buffer.
*/
func (tensorBackend *TensorBackend) Sigmoid(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "sigmoid", 0)
}

/*
Swish executes x * sigmoid(x) against a resident PJRT buffer.
*/
func (tensorBackend *TensorBackend) Swish(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "swish", 0)
}

/*
SELU executes scaled exponential linear unit against a resident PJRT buffer.
*/
func (tensorBackend *TensorBackend) SELU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "selu", 0)
}

/*
SwiGLU executes gated activation and halves the final dimension.
*/
func (tensorBackend *TensorBackend) SwiGLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	xlaInput, err := tensorBackend.require(input)

	if err != nil {
		return nil, err
	}

	outputShape, err := xlaSwiGLUOutputShape(xlaInput.shape)

	if err != nil {
		return nil, err
	}

	var output *C.XLA_Tensor

	if rc := C.xla_tensor_swiglu(xlaInput.handle, &output); rc != 0 || output == nil {
		return nil, fmt.Errorf("xla tensor: swiglu execution failed")
	}

	return newXLATensor(outputShape, output)
}

/*
Add executes elementwise addition against resident PJRT buffers.
*/
func (tensorBackend *TensorBackend) Add(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.binary(left, right, "add")
}

/*
Mul executes elementwise multiplication against resident PJRT buffers.
*/
func (tensorBackend *TensorBackend) Mul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.binary(left, right, "mul")
}

/*
Matmul executes row-major matrix multiplication against resident PJRT buffers.
*/
func (tensorBackend *TensorBackend) Matmul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	xlaLeft, xlaRight, outputShape, err := tensorBackend.matmulInputs(left, right)

	if err != nil {
		return nil, err
	}

	var output *C.XLA_Tensor

	if rc := C.xla_tensor_matmul(xlaLeft.handle, xlaRight.handle, &output); rc != 0 || output == nil {
		return nil, fmt.Errorf("xla tensor: matmul execution failed")
	}

	return newXLATensor(outputShape, output)
}

/*
MatmulAdd executes matrix multiplication plus bias against resident PJRT buffers.
*/
func (tensorBackend *TensorBackend) MatmulAdd(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.matmulAdd(left, right, bias, false)
}

/*
MatmulAddGELU executes fused matrix multiplication, bias, and GELU.
*/
func (tensorBackend *TensorBackend) MatmulAddGELU(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.matmulAdd(left, right, bias, true)
}

func (tensorBackend *TensorBackend) unary(
	input computetensor.Float64Tensor, name string, alpha float64,
) (computetensor.Float64Tensor, error) {
	xlaInput, err := tensorBackend.require(input)

	if err != nil {
		return nil, err
	}

	var output *C.XLA_Tensor
	var rc C.int

	switch name {
	case "relu":
		rc = C.xla_tensor_relu(xlaInput.handle, &output)
	case "leaky_relu":
		rc = C.xla_tensor_leaky_relu(xlaInput.handle, C.double(alpha), &output)
	case "gelu":
		rc = C.xla_tensor_gelu(xlaInput.handle, &output)
	case "tanh":
		rc = C.xla_tensor_tanh(xlaInput.handle, &output)
	case "sigmoid":
		rc = C.xla_tensor_sigmoid(xlaInput.handle, &output)
	case "swish":
		rc = C.xla_tensor_swish(xlaInput.handle, &output)
	case "selu":
		rc = C.xla_tensor_selu(xlaInput.handle, &output)
	default:
		return nil, fmt.Errorf("xla tensor: unknown unary kernel %q", name)
	}

	if rc != 0 || output == nil {
		return nil, fmt.Errorf("xla tensor: %s execution failed", name)
	}

	return newXLATensor(xlaInput.shape, output)
}

func (tensorBackend *TensorBackend) binary(
	left, right computetensor.Float64Tensor, name string,
) (computetensor.Float64Tensor, error) {
	xlaLeft, err := tensorBackend.require(left)

	if err != nil {
		return nil, err
	}

	xlaRight, err := tensorBackend.require(right)

	if err != nil {
		return nil, err
	}

	if !xlaLeft.shape.Equal(xlaRight.shape) {
		return nil, fmt.Errorf("xla tensor: binary operation shape mismatch")
	}

	var output *C.XLA_Tensor
	var rc C.int

	switch name {
	case "add":
		rc = C.xla_tensor_add(xlaLeft.handle, xlaRight.handle, &output)
	case "mul":
		rc = C.xla_tensor_mul(xlaLeft.handle, xlaRight.handle, &output)
	default:
		return nil, fmt.Errorf("xla tensor: unknown binary kernel %q", name)
	}

	if rc != 0 || output == nil {
		return nil, fmt.Errorf("xla tensor: %s execution failed", name)
	}

	return newXLATensor(xlaLeft.shape, output)
}

func (tensorBackend *TensorBackend) matmulAdd(
	left, right, bias computetensor.Float64Tensor, gelu bool,
) (computetensor.Float64Tensor, error) {
	xlaLeft, xlaRight, outputShape, err := tensorBackend.matmulInputs(left, right)

	if err != nil {
		return nil, err
	}

	xlaBias, err := tensorBackend.require(bias)

	if err != nil {
		return nil, err
	}

	outputDims := outputShape.Dims()
	biasLen := xlaBias.Len()

	if biasLen != outputDims[1] && biasLen != outputShape.Len() {
		return nil, fmt.Errorf(
			"xla tensor: fused matmul bias length %d must be N=%d or M*N=%d",
			biasLen, outputDims[1], outputShape.Len(),
		)
	}

	var output *C.XLA_Tensor

	rc := C.xla_tensor_matmul_add(
		xlaLeft.handle,
		xlaRight.handle,
		xlaBias.handle,
		&output,
		C.bool(gelu),
	)

	if rc != 0 || output == nil {
		return nil, fmt.Errorf("xla tensor: fused matmul execution failed")
	}

	return newXLATensor(outputShape, output)
}

func (tensorBackend *TensorBackend) matmulInputs(
	left, right computetensor.Float64Tensor,
) (*Tensor, *Tensor, computetensor.Shape, error) {
	xlaLeft, err := tensorBackend.require(left)

	if err != nil {
		return nil, nil, computetensor.Shape{}, err
	}

	xlaRight, err := tensorBackend.require(right)

	if err != nil {
		return nil, nil, computetensor.Shape{}, err
	}

	leftDims := xlaLeft.shape.Dims()
	rightDims := xlaRight.shape.Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 {
		return nil, nil, computetensor.Shape{}, fmt.Errorf("xla tensor: matmul requires rank-2 tensors")
	}

	if leftDims[1] != rightDims[0] {
		return nil, nil, computetensor.Shape{}, fmt.Errorf(
			"xla tensor: matmul dimension mismatch [%d,%d] x [%d,%d]",
			leftDims[0], leftDims[1], rightDims[0], rightDims[1],
		)
	}

	outputShape, err := computetensor.NewShape([]int{leftDims[0], rightDims[1]})

	if err != nil {
		return nil, nil, computetensor.Shape{}, err
	}

	return xlaLeft, xlaRight, outputShape, nil
}

func (tensorBackend *TensorBackend) require(
	input computetensor.Float64Tensor,
) (*Tensor, error) {
	if tensorBackend.closed.Load() != 0 {
		return nil, errors.New("xla tensor: backend is closed")
	}

	if input == nil {
		return nil, errors.New("xla tensor: nil input")
	}

	if input.Location() != computetensor.XLA {
		return nil, fmt.Errorf("xla tensor: cannot execute %s tensor", input.Location())
	}

	xlaInput, ok := input.(*Tensor)

	if !ok {
		return nil, fmt.Errorf("xla tensor: input is not owned by XLA backend")
	}

	if xlaInput.closed.Load() != 0 {
		return nil, errors.New("xla tensor: input is closed")
	}

	return xlaInput, nil
}

func newXLATensor(shape computetensor.Shape, handle *C.XLA_Tensor) (*Tensor, error) {
	bytes, err := shape.Bytes(computetensor.Float64)

	if err != nil {
		_ = C.xla_tensor_free(handle)

		return nil, err
	}

	return &Tensor{
		bytes:  bytes,
		shape:  shape,
		handle: handle,
	}, nil
}

func cShapeDims(shape computetensor.Shape) []C.int64_t {
	dims := shape.Dims()
	cDims := make([]C.int64_t, len(dims))

	for dimensionIndex, dimension := range dims {
		cDims[dimensionIndex] = C.int64_t(dimension)
	}

	return cDims
}

func xlaSwiGLUOutputShape(shape computetensor.Shape) (computetensor.Shape, error) {
	dimsCopy := slices.Clone(shape.Dims())

	if len(dimsCopy) == 0 {
		return computetensor.Shape{}, fmt.Errorf("xla tensor: swiglu requires at least one dimension")
	}

	lastDimensionIndex := len(dimsCopy) - 1

	if dimsCopy[lastDimensionIndex]%2 != 0 {
		return computetensor.Shape{}, fmt.Errorf("xla tensor: swiglu final dimension must be even")
	}

	dimsCopy[lastDimensionIndex] /= 2

	return computetensor.NewShape(dimsCopy)
}

/*
Tensor is persistent PJRT buffer storage.
*/
type Tensor struct {
	bytes  int
	shape  computetensor.Shape
	handle *C.XLA_Tensor
	closed atomic.Uint32
}

/*
Shape returns validated tensor dimensions.
*/
func (tensor *Tensor) Shape() computetensor.Shape {
	return tensor.shape
}

/*
DType reports float64 storage.
*/
func (tensor *Tensor) DType() computetensor.DType {
	return computetensor.Float64
}

/*
Location reports XLA PJRT ownership.
*/
func (tensor *Tensor) Location() computetensor.Location {
	return computetensor.XLA
}

/*
Len returns the number of scalar elements.
*/
func (tensor *Tensor) Len() int {
	return tensor.shape.Len()
}

/*
Bytes returns the storage footprint.
*/
func (tensor *Tensor) Bytes() int {
	return tensor.bytes
}

/*
CloneFloat64 downloads tensor contents into a fresh host slice.
*/
func (tensor *Tensor) CloneFloat64() ([]float64, error) {
	if tensor.closed.Load() != 0 {
		return nil, errors.New("xla tensor: tensor is closed")
	}

	values := make([]float64, tensor.Len())

	if len(values) == 0 {
		return values, nil
	}

	rc := C.xla_tensor_download_f64(
		tensor.handle,
		(*C.double)(unsafe.Pointer(&values[0])),
		C.int64_t(len(values)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla tensor: download failed")
	}

	return values, nil
}

/*
Close releases the PJRT buffer.
*/
func (tensor *Tensor) Close() error {
	if !tensor.closed.CompareAndSwap(0, 1) {
		return nil
	}

	if tensor.handle == nil {
		return nil
	}

	if rc := C.xla_tensor_free(tensor.handle); rc != 0 {
		return fmt.Errorf("xla tensor: close failed")
	}

	tensor.handle = nil

	return nil
}
