//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "metal_kernel_math.h"
import "C"

import (
	"fmt"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

// MathOps dispatches math kernels to the GPU via Metal.
// metallib must be the absolute path to math.metallib compiled from math.metal.
type MathOps struct {
	metallib string
}

// NewMathOps creates and initializes a MathOps instance.
func NewMathOps(metallib string) (*MathOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))
	if rc := C.metal_math_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_math_init failed (rc=%d): check %q exists", rc, metallib)
	}
	return &MathOps{metallib: metallib}, nil
}

// ---------------------------------------------------------------------------
// Matmul — shape=[M,K,N], data[0]=A [M*K], data[1]=B [K*N]
// ---------------------------------------------------------------------------

func (m *MathOps) Matmul(shape []int, data ...[]float64) ([]float64, error) {
	M, K, N := shape[0], shape[1], shape[2]
	A := toFloat32(data[0])
	B := toFloat32(data[1])
	C_ := make([]float32, M*N)
	rc := C.metal_matmul(
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_[0])),
		C.int(M), C.int(K), C.int(N),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_matmul failed (rc=%d)", rc)
	}
	return toFloat64(C_), nil
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

func (m *MathOps) Add(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("metal_add: two input buffers are required")
	}

	n := len(data[0])
	if n == 0 {
		return []float64{}, nil
	}

	if len(data[1]) != n {
		return nil, fmt.Errorf("metal_add: input length mismatch %d != %d", n, len(data[1]))
	}

	a := toFloat32(data[0])
	b := toFloat32(data[1])
	out := make([]float32, n)
	rc := C.metal_add(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_add failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}

// ---------------------------------------------------------------------------
// Mul
// ---------------------------------------------------------------------------

func (m *MathOps) Mul(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("metal_mul: two input buffers are required")
	}

	n := len(data[0])
	if n == 0 {
		return []float64{}, nil
	}

	if len(data[1]) != n {
		return nil, fmt.Errorf("metal_mul: input length mismatch %d != %d", n, len(data[1]))
	}

	a := toFloat32(data[0])
	b := toFloat32(data[1])
	out := make([]float32, n)
	rc := C.metal_mul(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_mul failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}

/*
AddTensor performs resident Metal elementwise addition.
Left and right must have identical shapes; broadcasting is not supported.
*/
func (m *MathOps) AddTensor(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return m.binaryTensor(left, right, "add")
}

/*
MulTensor performs resident Metal elementwise multiplication.
Left and right must have identical shapes; broadcasting is not supported.
*/
func (m *MathOps) MulTensor(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return m.binaryTensor(left, right, "mul")
}

/*
MatmulTensor performs resident Metal row-major matrix multiplication.
*/
func (m *MathOps) MatmulTensor(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	metalLeft, err := requireMetalTensor(left)

	if err != nil {
		return nil, err
	}

	metalRight, err := requireMetalTensor(right)

	if err != nil {
		return nil, err
	}

	leftDims := metalLeft.shape.Dims()
	rightDims := metalRight.shape.Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 {
		return nil, fmt.Errorf("metal tensor: matmul requires rank-2 tensors")
	}

	if leftDims[1] != rightDims[0] {
		return nil, fmt.Errorf(
			"metal tensor: matmul dimension mismatch [%d,%d] x [%d,%d]",
			leftDims[0], leftDims[1], rightDims[0], rightDims[1],
		)
	}

	outputShape, err := computetensor.NewShape([]int{leftDims[0], rightDims[1]})

	if err != nil {
		return nil, err
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_matmul_tensor(
		metalLeft.buffer,
		metalRight.buffer,
		output.buffer,
		C.int(leftDims[0]),
		C.int(leftDims[1]),
		C.int(rightDims[1]),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: matmul launch failed")
	}

	return output, nil
}

/*
MatmulAddTensor performs resident Metal matrix multiplication with broadcast bias.
*/
func (m *MathOps) MatmulAddTensor(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return m.matmulAddTensor(left, right, bias, false)
}

/*
MatmulAddGELUTensor performs resident Metal matrix multiplication, bias, and GELU.
*/
func (m *MathOps) MatmulAddGELUTensor(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return m.matmulAddTensor(left, right, bias, true)
}

func (m *MathOps) MatmulFlatTensor(
	left, right computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	M, K, N int,
) (computetensor.Float64Tensor, error) {
	metalLeft, err := requireMetalTensor(left)

	if err != nil {
		return nil, err
	}

	metalRight, err := requireMetalTensor(right)

	if err != nil {
		return nil, err
	}

	if M <= 0 || K <= 0 || N <= 0 {
		return nil, fmt.Errorf("metal tensor: flat matmul dimensions must be positive")
	}

	if metalLeft.Len() != M*K || metalRight.Len() != K*N {
		return nil, fmt.Errorf(
			"metal tensor: flat matmul length mismatch A=%d want %d B=%d want %d",
			metalLeft.Len(),
			M*K,
			metalRight.Len(),
			K*N,
		)
	}

	if outputShape.Len() != M*N {
		return nil, fmt.Errorf(
			"metal tensor: flat matmul output shape length %d must equal M*N=%d",
			outputShape.Len(),
			M*N,
		)
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_matmul_tensor(
		metalLeft.buffer,
		metalRight.buffer,
		output.buffer,
		C.int(M),
		C.int(K),
		C.int(N),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: flat matmul launch failed")
	}

	return output, nil
}

func (m *MathOps) MatmulAddFlatTensor(
	left, right, bias computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	M, K, N int,
) (computetensor.Float64Tensor, error) {
	metalLeft, err := requireMetalTensor(left)

	if err != nil {
		return nil, err
	}

	metalRight, err := requireMetalTensor(right)

	if err != nil {
		return nil, err
	}

	metalBias, err := requireMetalTensor(bias)

	if err != nil {
		return nil, err
	}

	if M <= 0 || K <= 0 || N <= 0 {
		return nil, fmt.Errorf("metal tensor: flat fused matmul dimensions must be positive")
	}

	if metalLeft.Len() != M*K || metalRight.Len() != K*N {
		return nil, fmt.Errorf(
			"metal tensor: flat fused matmul length mismatch A=%d want %d B=%d want %d",
			metalLeft.Len(),
			M*K,
			metalRight.Len(),
			K*N,
		)
	}

	if metalBias.Len() != N && metalBias.Len() != M*N {
		return nil, fmt.Errorf(
			"metal tensor: flat fused matmul bias length %d must be N=%d or M*N=%d",
			metalBias.Len(),
			N,
			M*N,
		)
	}

	if outputShape.Len() != M*N {
		return nil, fmt.Errorf(
			"metal tensor: flat fused matmul output shape length %d must equal M*N=%d",
			outputShape.Len(),
			M*N,
		)
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_matmul_add_tensor(
		metalLeft.buffer,
		metalRight.buffer,
		metalBias.buffer,
		output.buffer,
		C.int(M),
		C.int(K),
		C.int(N),
		C.int(metalBias.Len()),
		C.int(0),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: flat fused matmul launch failed")
	}

	return output, nil
}

func (m *MathOps) LayerNormTensor(
	input, weight, bias computetensor.Float64Tensor,
	eps float64,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	metalWeight, err := requireMetalTensor(weight)

	if err != nil {
		return nil, err
	}

	metalBias, err := requireMetalTensor(bias)

	if err != nil {
		return nil, err
	}

	dimensions := metalInput.shape.Dims()

	if len(dimensions) == 0 {
		return nil, fmt.Errorf("metal tensor: layernorm input shape is required")
	}

	dModel := dimensions[len(dimensions)-1]

	if dModel <= 0 || metalInput.Len()%dModel != 0 {
		return nil, fmt.Errorf("metal tensor: invalid layernorm final dimension %d", dModel)
	}

	if metalWeight.Len() != dModel || metalBias.Len() != dModel {
		return nil, fmt.Errorf(
			"metal tensor: layernorm weight/bias length must equal d_model=%d",
			dModel,
		)
	}

	output, err := newMetalTensor(metalInput.shape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_layernorm_tensor(
		metalInput.buffer,
		output.buffer,
		metalWeight.buffer,
		metalBias.buffer,
		C.int(metalInput.Len()/dModel),
		C.int(dModel),
		C.float(eps),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: layernorm launch failed")
	}

	return output, nil
}

func (m *MathOps) RMSNormTensor(
	input, weight computetensor.Float64Tensor,
	eps float64,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	metalWeight, err := requireMetalTensor(weight)

	if err != nil {
		return nil, err
	}

	dimensions := metalInput.shape.Dims()

	if len(dimensions) == 0 {
		return nil, fmt.Errorf("metal tensor: rmsnorm input shape is required")
	}

	dModel := dimensions[len(dimensions)-1]

	if dModel <= 0 || metalInput.Len()%dModel != 0 {
		return nil, fmt.Errorf("metal tensor: invalid rmsnorm final dimension %d", dModel)
	}

	if metalWeight.Len() != dModel {
		return nil, fmt.Errorf("metal tensor: rmsnorm weight length must equal d_model=%d", dModel)
	}

	output, err := newMetalTensor(metalInput.shape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_rmsnorm_tensor(
		metalInput.buffer,
		output.buffer,
		metalWeight.buffer,
		C.int(metalInput.Len()/dModel),
		C.int(dModel),
		C.float(eps),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: rmsnorm launch failed")
	}

	return output, nil
}

func (m *MathOps) binaryTensor(
	left, right computetensor.Float64Tensor, name string,
) (computetensor.Float64Tensor, error) {
	metalLeft, err := requireMetalTensor(left)

	if err != nil {
		return nil, err
	}

	metalRight, err := requireMetalTensor(right)

	if err != nil {
		return nil, err
	}

	if !metalLeft.shape.Equal(metalRight.shape) {
		return nil, fmt.Errorf("metal tensor: binary operation shape mismatch")
	}

	output, err := newMetalTensor(metalLeft.shape)

	if err != nil {
		return nil, err
	}

	var rc C.int

	switch name {
	case "add":
		rc = C.metal_add_tensor(metalLeft.buffer, metalRight.buffer, output.buffer, C.int(output.Len()))
	case "mul":
		rc = C.metal_mul_tensor(metalLeft.buffer, metalRight.buffer, output.buffer, C.int(output.Len()))
	default:
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: unknown binary kernel %q", name)
	}

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: %s launch failed", name)
	}

	return output, nil
}

func (m *MathOps) matmulAddTensor(
	left, right, bias computetensor.Float64Tensor, gelu bool,
) (computetensor.Float64Tensor, error) {
	metalLeft, metalRight, metalBias, outputShape, err := metalMatmulAddInputs(
		left, right, bias,
	)

	if err != nil {
		return nil, err
	}

	leftDims := metalLeft.shape.Dims()
	rightDims := metalRight.shape.Dims()
	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	applyGELU := 0

	if gelu {
		applyGELU = 1
	}

	rc := C.metal_matmul_add_tensor(
		metalLeft.buffer,
		metalRight.buffer,
		metalBias.buffer,
		output.buffer,
		C.int(leftDims[0]),
		C.int(leftDims[1]),
		C.int(rightDims[1]),
		C.int(metalBias.Len()),
		C.int(applyGELU),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: fused matmul launch failed")
	}

	return output, nil
}

func metalMatmulAddInputs(
	left, right, bias computetensor.Float64Tensor,
) (*Tensor, *Tensor, *Tensor, computetensor.Shape, error) {
	metalLeft, err := requireMetalTensor(left)

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	metalRight, err := requireMetalTensor(right)

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	metalBias, err := requireMetalTensor(bias)

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	leftDims := metalLeft.shape.Dims()
	rightDims := metalRight.shape.Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 {
		return nil, nil, nil, computetensor.Shape{}, fmt.Errorf("metal tensor: fused matmul requires rank-2 tensors")
	}

	if leftDims[1] != rightDims[0] {
		return nil, nil, nil, computetensor.Shape{}, fmt.Errorf(
			"metal tensor: fused matmul dimension mismatch [%d,%d] x [%d,%d]",
			leftDims[0], leftDims[1], rightDims[0], rightDims[1],
		)
	}

	M, N := leftDims[0], rightDims[1]
	biasLen := metalBias.Len()

	if biasLen != N && biasLen != M*N {
		return nil, nil, nil, computetensor.Shape{}, fmt.Errorf(
			"metal tensor: fused matmul bias length %d must be N=%d or M*N=%d",
			biasLen, N, M*N,
		)
	}

	outputShape, err := computetensor.NewShape([]int{M, N})

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	return metalLeft, metalRight, metalBias, outputShape, nil
}

// ---------------------------------------------------------------------------
// InvSqrtDimScale — shape[-1] is the dim
// ---------------------------------------------------------------------------

func (m *MathOps) InvSqrtDimScale(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dim := shape[len(shape)-1]
	src := toFloat32(data[0])
	dst := make([]float32, n)
	rc := C.metal_inv_sqrt_dim_scale(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n), C.int(dim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_inv_sqrt_dim_scale failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ---------------------------------------------------------------------------
// Exp
// ---------------------------------------------------------------------------

func (m *MathOps) Exp(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("metal_exp: input[0] is required")
	}

	n := len(data[0])
	if n == 0 {
		return []float64{}, nil
	}

	src := toFloat32(data[0])
	dst := make([]float32, n)
	rc := C.metal_exp(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_exp failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------

func (m *MathOps) Log(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("metal_log: input[0] is required")
	}

	n := len(data[0])
	if n == 0 {
		return []float64{}, nil
	}

	src := toFloat32(data[0])
	dst := make([]float32, n)
	rc := C.metal_log(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_log failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ---------------------------------------------------------------------------
// Softmax — shape=[..., dim_size]
// ---------------------------------------------------------------------------

func (m *MathOps) Softmax(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("metal_softmax: shape is required")
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("metal_softmax: input[0] is required")
	}

	dimSize := shape[len(shape)-1]
	if dimSize <= 0 {
		return nil, fmt.Errorf("metal_softmax: dim size must be positive, got %d", dimSize)
	}

	n := len(data[0])
	if n == 0 {
		return []float64{}, nil
	}

	if n%dimSize != 0 {
		return nil, fmt.Errorf("metal_softmax: input length %d not divisible by dim size %d", n, dimSize)
	}

	numRows := n / dimSize
	src := toFloat32(data[0])
	dst := make([]float32, n)
	rc := C.metal_softmax(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(numRows), C.int(dimSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_softmax failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// LogSumExp computes log(sum(exp(x))) over the last dimension.
func (m *MathOps) LogSumExp(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("metal_logsumexp: shape is required")
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("metal_logsumexp: input[0] is required")
	}

	dimSize := shape[len(shape)-1]
	if dimSize <= 0 {
		return nil, fmt.Errorf("metal_logsumexp: dim size must be positive, got %d", dimSize)
	}

	n := len(data[0])
	if n == 0 {
		return nil, fmt.Errorf("metal_logsumexp: input[0] must be non-empty")
	}

	if n%dimSize != 0 {
		return nil, fmt.Errorf("metal_logsumexp: input length %d not divisible by dim size %d", n, dimSize)
	}

	numRows := n / dimSize
	src := toFloat32(data[0])
	dst := make([]float32, numRows)
	rc := C.metal_logsumexp(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(numRows), C.int(dimSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_logsumexp failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// Dropout applies inverted dropout using a stateless index/step keyed mask.
func (m *MathOps) Dropout(
	probability float64,
	training bool,
	seed int,
	data []float64,
) ([]float64, error) {
	if probability < 0.0 || probability > 1.0 {
		return nil, fmt.Errorf("invalid probability: %v, must be in [0,1]", probability)
	}

	n := len(data)
	if n == 0 {
		return []float64{}, nil
	}

	src := toFloat32(data)
	dst := make([]float32, n)
	trainingInt := 0
	if training {
		trainingInt = 1
	}
	rc := C.metal_dropout(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n), C.float(probability), C.int(trainingInt), C.int(seed),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_dropout failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

func (m *MathOps) LayerNorm(shape []int, eps float64, weight, bias []float64, data ...[]float64) ([]float64, error) {
	dModel := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dModel
	src := toFloat32(data[0])
	dst := make([]float32, n)
	w := toFloat32(weight)
	b := toFloat32(bias)
	rc := C.metal_layernorm(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.int(numRows), C.int(dModel), C.float(eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_layernorm failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

func (m *MathOps) RMSNorm(shape []int, eps float64, weight []float64, data ...[]float64) ([]float64, error) {
	dModel := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dModel
	src := toFloat32(data[0])
	dst := make([]float32, n)
	w := toFloat32(weight)
	rc := C.metal_rmsnorm(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		C.int(numRows), C.int(dModel), C.float(eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_rmsnorm failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// Sign: elementwise sign
func (m *MathOps) Sign(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	src := toFloat32(data[0])
	dst := make([]float32, n)
	rc := C.metal_sign(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_sign failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// Outer: outer product a[M] x b[N] → dst[M*N]
func (m *MathOps) Outer(shape []int, data ...[]float64) ([]float64, error) {
	M, N := shape[0], shape[1]
	a := toFloat32(data[0])
	b := toFloat32(data[1])
	dst := make([]float32, M*N)
	rc := C.metal_outer(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(N),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_outer failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}
