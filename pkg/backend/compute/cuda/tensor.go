//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "activation.h"
// #include "math.h"
// #include "tensor.h"
import "C"

import (
	"errors"
	"fmt"
	"sync/atomic"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ computetensor.Backend = (*TensorBackend)(nil)

/*
TensorBackend owns CUDA device tensors.
*/
type TensorBackend struct {
	closed atomic.Uint32
}

/*
NewTensorBackend creates a CUDA resident tensor backend.
*/
func NewTensorBackend() *TensorBackend {
	return &TensorBackend{}
}

/*
Location identifies CUDA storage ownership.
*/
func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.CUDA
}

/*
UploadFloat64 copies host values into CUDA device memory.
*/
func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	if tensorBackend.closed.Load() != 0 {
		return nil, errors.New("cuda tensor: backend is closed")
	}

	if !shape.Valid() {
		return nil, errors.New("cuda tensor: invalid shape")
	}

	if shape.Len() != len(values) {
		return nil, fmt.Errorf(
			"cuda tensor: shape has %d elements but upload received %d values",
			shape.Len(), len(values),
		)
	}

	deviceTensor, err := tensorBackend.empty(shape)

	if err != nil {
		return nil, err
	}

	if len(values) == 0 {
		return deviceTensor, nil
	}

	rc := C.cuda_tensor_upload_double(
		deviceTensor.device,
		(*C.double)(unsafe.Pointer(&values[0])),
		C.size_t(len(values)),
	)

	if rc != 0 {
		_ = deviceTensor.Close()

		return nil, fmt.Errorf("cuda tensor: upload failed")
	}

	return deviceTensor, nil
}

/*
DownloadFloat64 copies CUDA device memory back into host values.
*/
func (tensorBackend *TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	if tensorBackend.closed.Load() != 0 {
		return nil, errors.New("cuda tensor: backend is closed")
	}

	if input == nil {
		return nil, errors.New("cuda tensor: nil input")
	}

	if input.Location() != computetensor.CUDA {
		return nil, fmt.Errorf("cuda tensor: cannot download %s tensor", input.Location())
	}

	return input.CloneFloat64()
}

/*
Close releases the backend.
*/
func (tensorBackend *TensorBackend) Close() error {
	tensorBackend.closed.Store(1)

	return nil
}

/*
ReLU launches a device-to-device CUDA ReLU kernel.
*/
func (tensorBackend *TensorBackend) ReLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "relu", 0)
}

/*
LeakyReLU launches a device-to-device CUDA leaky ReLU kernel.
*/
func (tensorBackend *TensorBackend) LeakyReLU(
	input computetensor.Float64Tensor, alpha float64,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "leaky_relu", alpha)
}

/*
GELU launches a device-to-device CUDA GELU kernel.
*/
func (tensorBackend *TensorBackend) GELU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "gelu", 0)
}

/*
Tanh launches a device-to-device CUDA tanh kernel.
*/
func (tensorBackend *TensorBackend) Tanh(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "tanh", 0)
}

/*
Sigmoid launches a device-to-device CUDA sigmoid kernel.
*/
func (tensorBackend *TensorBackend) Sigmoid(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "sigmoid", 0)
}

/*
Swish launches a device-to-device CUDA Swish kernel.
*/
func (tensorBackend *TensorBackend) Swish(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "swish", 0)
}

/*
SELU launches a device-to-device CUDA SELU kernel.
*/
func (tensorBackend *TensorBackend) SELU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.unary(input, "selu", 0)
}

/*
SwiGLU launches a device-to-device CUDA SwiGLU kernel.
*/
func (tensorBackend *TensorBackend) SwiGLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	deviceInput, err := tensorBackend.require(input)

	if err != nil {
		return nil, err
	}

	outputShape, err := cudaSwiGLUOutputShape(deviceInput.shape)

	if err != nil {
		return nil, err
	}

	output, err := tensorBackend.empty(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.cuda_swiglu_device((*C.double)(deviceInput.device), (*C.double)(output.device), C.int(outputShape.Len()))

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("cuda tensor: swiglu launch failed")
	}

	return output, nil
}

/*
Add launches a device-to-device CUDA add kernel.
*/
func (tensorBackend *TensorBackend) Add(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.binary(left, right, "add")
}

/*
Mul launches a device-to-device CUDA multiply kernel.
*/
func (tensorBackend *TensorBackend) Mul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.binary(left, right, "mul")
}

/*
Matmul launches a device-to-device CUDA matrix multiplication kernel.
*/
func (tensorBackend *TensorBackend) Matmul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	deviceLeft, err := tensorBackend.require(left)

	if err != nil {
		return nil, err
	}

	deviceRight, err := tensorBackend.require(right)

	if err != nil {
		return nil, err
	}

	leftDims := deviceLeft.shape.Dims()
	rightDims := deviceRight.shape.Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 {
		return nil, fmt.Errorf("cuda tensor: matmul requires rank-2 tensors")
	}

	if leftDims[1] != rightDims[0] {
		return nil, fmt.Errorf(
			"cuda tensor: matmul dimension mismatch [%d,%d] x [%d,%d]",
			leftDims[0], leftDims[1], rightDims[0], rightDims[1],
		)
	}

	outputShape, err := computetensor.NewShape([]int{leftDims[0], rightDims[1]})

	if err != nil {
		return nil, err
	}

	output, err := tensorBackend.empty(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.cuda_matmul_device(
		(*C.double)(deviceLeft.device),
		(*C.double)(deviceRight.device),
		(*C.double)(output.device),
		C.int(leftDims[0]),
		C.int(leftDims[1]),
		C.int(rightDims[1]),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("cuda tensor: matmul launch failed")
	}

	return output, nil
}

/*
MatmulAdd launches a fused CUDA matrix multiplication and bias kernel.
*/
func (tensorBackend *TensorBackend) MatmulAdd(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.matmulAdd(left, right, bias, false)
}

/*
MatmulAddGELU launches a fused CUDA matrix multiplication, bias, and GELU kernel.
*/
func (tensorBackend *TensorBackend) MatmulAddGELU(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.matmulAdd(left, right, bias, true)
}

func (tensorBackend *TensorBackend) unary(
	input computetensor.Float64Tensor, name string, alpha float64,
) (computetensor.Float64Tensor, error) {
	deviceInput, err := tensorBackend.require(input)

	if err != nil {
		return nil, err
	}

	output, err := tensorBackend.empty(deviceInput.shape)

	if err != nil {
		return nil, err
	}

	var rc C.int

	switch name {
	case "relu":
		rc = C.cuda_relu_device((*C.double)(deviceInput.device), (*C.double)(output.device), C.int(deviceInput.Len()))
	case "leaky_relu":
		rc = C.cuda_leaky_relu_device(
			(*C.double)(deviceInput.device),
			(*C.double)(output.device),
			C.double(alpha),
			C.int(deviceInput.Len()),
		)
	case "gelu":
		rc = C.cuda_gelu_device((*C.double)(deviceInput.device), (*C.double)(output.device), C.int(deviceInput.Len()))
	case "tanh":
		rc = C.cuda_tanh_device((*C.double)(deviceInput.device), (*C.double)(output.device), C.int(deviceInput.Len()))
	case "sigmoid":
		rc = C.cuda_sigmoid_device((*C.double)(deviceInput.device), (*C.double)(output.device), C.int(deviceInput.Len()))
	case "swish":
		rc = C.cuda_swish_device((*C.double)(deviceInput.device), (*C.double)(output.device), C.int(deviceInput.Len()))
	case "selu":
		rc = C.cuda_selu_device((*C.double)(deviceInput.device), (*C.double)(output.device), C.int(deviceInput.Len()))
	default:
		_ = output.Close()

		return nil, fmt.Errorf("cuda tensor: unknown unary kernel %q", name)
	}

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("cuda tensor: %s launch failed", name)
	}

	return output, nil
}

func (tensorBackend *TensorBackend) matmulAdd(
	left, right, bias computetensor.Float64Tensor, gelu bool,
) (computetensor.Float64Tensor, error) {
	deviceLeft, deviceRight, deviceBias, outputShape, err := tensorBackend.matmulAddInputs(
		left, right, bias,
	)

	if err != nil {
		return nil, err
	}

	leftDims := deviceLeft.shape.Dims()
	rightDims := deviceRight.shape.Dims()
	output, err := tensorBackend.empty(outputShape)

	if err != nil {
		return nil, err
	}

	applyGELU := 0

	if gelu {
		applyGELU = 1
	}

	rc := C.cuda_matmul_add_device(
		(*C.double)(deviceLeft.device),
		(*C.double)(deviceRight.device),
		(*C.double)(deviceBias.device),
		(*C.double)(output.device),
		C.int(leftDims[0]),
		C.int(leftDims[1]),
		C.int(rightDims[1]),
		C.int(deviceBias.Len()),
		C.int(applyGELU),
		C.int(1), // sync_device: wait for kernel before returning (see cuda_matmul_add_device)
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("cuda tensor: fused matmul launch failed")
	}

	return output, nil
}

func (tensorBackend *TensorBackend) matmulAddInputs(
	left, right, bias computetensor.Float64Tensor,
) (*Tensor, *Tensor, *Tensor, computetensor.Shape, error) {
	deviceLeft, err := tensorBackend.require(left)

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	deviceRight, err := tensorBackend.require(right)

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	deviceBias, err := tensorBackend.require(bias)

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	leftDims := deviceLeft.shape.Dims()
	rightDims := deviceRight.shape.Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 {
		return nil, nil, nil, computetensor.Shape{}, fmt.Errorf("cuda tensor: fused matmul requires rank-2 tensors")
	}

	if leftDims[1] != rightDims[0] {
		return nil, nil, nil, computetensor.Shape{}, fmt.Errorf(
			"cuda tensor: fused matmul dimension mismatch [%d,%d] x [%d,%d]",
			leftDims[0], leftDims[1], rightDims[0], rightDims[1],
		)
	}

	M, N := leftDims[0], rightDims[1]
	biasLen := deviceBias.Len()

	if biasLen != N && biasLen != M*N {
		return nil, nil, nil, computetensor.Shape{}, fmt.Errorf(
			"cuda tensor: fused matmul bias length %d must be N=%d or M*N=%d",
			biasLen, N, M*N,
		)
	}

	outputShape, err := computetensor.NewShape([]int{M, N})

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	return deviceLeft, deviceRight, deviceBias, outputShape, nil
}

func (tensorBackend *TensorBackend) binary(
	left, right computetensor.Float64Tensor, name string,
) (computetensor.Float64Tensor, error) {
	deviceLeft, err := tensorBackend.require(left)

	if err != nil {
		return nil, err
	}

	deviceRight, err := tensorBackend.require(right)

	if err != nil {
		return nil, err
	}

	if !deviceLeft.shape.Equal(deviceRight.shape) {
		return nil, fmt.Errorf("cuda tensor: binary operation shape mismatch")
	}

	output, err := tensorBackend.empty(deviceLeft.shape)

	if err != nil {
		return nil, err
	}

	var rc C.int

	switch name {
	case "add":
		rc = C.cuda_add_device(
			(*C.double)(deviceLeft.device),
			(*C.double)(deviceRight.device),
			(*C.double)(output.device),
			C.int(output.Len()),
		)
	case "mul":
		rc = C.cuda_mul_device(
			(*C.double)(deviceLeft.device),
			(*C.double)(deviceRight.device),
			(*C.double)(output.device),
			C.int(output.Len()),
		)
	default:
		_ = output.Close()

		return nil, fmt.Errorf("cuda tensor: unknown binary kernel %q", name)
	}

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("cuda tensor: %s launch failed", name)
	}

	return output, nil
}

func (tensorBackend *TensorBackend) require(
	input computetensor.Float64Tensor,
) (*Tensor, error) {
	if tensorBackend.closed.Load() != 0 {
		return nil, errors.New("cuda tensor: backend is closed")
	}

	if input == nil {
		return nil, errors.New("cuda tensor: nil input")
	}

	if input.Location() != computetensor.CUDA {
		return nil, fmt.Errorf("cuda tensor: cannot execute %s tensor", input.Location())
	}

	deviceInput, ok := input.(*Tensor)

	if !ok {
		return nil, fmt.Errorf("cuda tensor: input is not owned by CUDA backend")
	}

	if deviceInput.closed.Load() != 0 {
		return nil, errors.New("cuda tensor: input is closed")
	}

	return deviceInput, nil
}

func (tensorBackend *TensorBackend) empty(shape computetensor.Shape) (*Tensor, error) {
	bytes, err := shape.Bytes(computetensor.Float64)

	if err != nil {
		return nil, err
	}

	var device unsafe.Pointer

	if bytes > 0 {
		device = C.cuda_tensor_alloc(C.size_t(bytes))

		if device == nil {
			return nil, fmt.Errorf("cuda tensor: allocation of %d bytes failed", bytes)
		}
	}

	return &Tensor{
		bytes:  bytes,
		shape:  shape,
		device: device,
	}, nil
}

/*
Tensor is persistent CUDA device storage.
*/
type Tensor struct {
	bytes  int
	shape  computetensor.Shape
	device unsafe.Pointer
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
Location reports CUDA device ownership.
*/
func (tensor *Tensor) Location() computetensor.Location {
	return computetensor.CUDA
}

/*
Len reports the number of tensor elements.
*/
func (tensor *Tensor) Len() int {
	return tensor.shape.Len()
}

/*
Bytes reports the CUDA allocation size.
*/
func (tensor *Tensor) Bytes() int {
	return tensor.bytes
}

/*
CloneFloat64 downloads the CUDA tensor to host memory.
*/
func (tensor *Tensor) CloneFloat64() ([]float64, error) {
	if tensor.closed.Load() != 0 {
		return nil, errors.New("cuda tensor: tensor is closed")
	}

	values := make([]float64, tensor.Len())

	if len(values) == 0 {
		return values, nil
	}

	rc := C.cuda_tensor_download_double(
		tensor.device,
		(*C.double)(unsafe.Pointer(&values[0])),
		C.size_t(len(values)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda tensor: download failed")
	}

	return values, nil
}

/*
Close releases CUDA device memory.
*/
func (tensor *Tensor) Close() error {
	if !tensor.closed.CompareAndSwap(0, 1) {
		return nil
	}

	devicePtr := tensor.device
	tensor.bytes = 0
	tensor.device = nil

	var rc C.int

	if devicePtr != nil {
		rc = C.cuda_tensor_free(devicePtr)
	}

	if rc != 0 {
		return fmt.Errorf("cuda tensor: free failed")
	}

	return nil
}

func cudaSwiGLUOutputShape(shape computetensor.Shape) (computetensor.Shape, error) {
	dimsCopy := append([]int(nil), shape.Dims()...)

	if shape.Len()%2 != 0 {
		return computetensor.Shape{}, fmt.Errorf("cuda tensor: swiglu input length must be even")
	}

	if len(dimsCopy) == 0 {
		return computetensor.NewShape([]int{shape.Len() / 2})
	}

	lastIndex := len(dimsCopy) - 1

	if dimsCopy[lastIndex]%2 != 0 {
		return computetensor.Shape{}, fmt.Errorf("cuda tensor: swiglu final dimension must be even")
	}

	dimsCopy[lastIndex] /= 2

	return computetensor.NewShape(dimsCopy)
}
