//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "math.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// CUDAMathOps dispatches math kernels to the GPU via CUDA.
type CUDAMathOps struct{}

// NewMathOps creates a CUDAMathOps.
func NewMathOps() *CUDAMathOps { return &CUDAMathOps{} }

// Matmul: shape=[M,K,N], data[0]=A, data[1]=B
func (c *CUDAMathOps) Matmul(shape []int, data ...[]float64) ([]float64, error) {
	M, K, N := shape[0], shape[1], shape[2]
	dst := make([]float64, M*N)
	rc := C.cuda_matmul(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(K), C.int(N),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_matmul failed (rc=%d)", rc)
	}
	return dst, nil
}

// Add: elementwise data[0]+data[1]
func (c *CUDAMathOps) Add(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.cuda_add(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_add failed (rc=%d)", rc)
	}
	return dst, nil
}

// Mul: elementwise data[0]*data[1]
func (c *CUDAMathOps) Mul(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.cuda_mul(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_mul failed (rc=%d)", rc)
	}
	return dst, nil
}

// InvSqrtDimScale: data[0] * (1/sqrt(shape[-1]))
func (c *CUDAMathOps) InvSqrtDimScale(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dim := shape[len(shape)-1]
	dst := make([]float64, n)
	rc := C.cuda_inv_sqrt_dim_scale(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n), C.int(dim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_inv_sqrt_dim_scale failed (rc=%d)", rc)
	}
	return dst, nil
}

// Exp: elementwise exp
func (c *CUDAMathOps) Exp(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.cuda_exp(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_exp failed (rc=%d)", rc)
	}
	return dst, nil
}

// Log: elementwise log
func (c *CUDAMathOps) Log(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.cuda_log(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_log failed (rc=%d)", rc)
	}
	return dst, nil
}

// Softmax: shape=[...,dim_size], softmax over last dim
func (c *CUDAMathOps) Softmax(shape []int, data ...[]float64) ([]float64, error) {
	dimSize := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dimSize
	dst := make([]float64, n)
	rc := C.cuda_softmax(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(numRows), C.int(dimSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_softmax failed (rc=%d)", rc)
	}
	return dst, nil
}

// LogSumExp computes log(sum(exp(x))) over the last dimension.
func (c *CUDAMathOps) LogSumExp(shape []int, data ...[]float64) ([]float64, error) {
	dimSize := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dimSize
	dst := make([]float64, numRows)
	rc := C.cuda_logsumexp(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(numRows), C.int(dimSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_logsumexp failed (rc=%d)", rc)
	}
	return dst, nil
}

// Dropout applies inverted dropout using a stateless index/step keyed mask.
func (c *CUDAMathOps) Dropout(
	probability float64,
	training bool,
	seed int,
	data []float64,
) ([]float64, error) {
	n := len(data)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	trainingInt := 0
	if training {
		trainingInt = 1
	}
	rc := C.cuda_dropout(
		(*C.double)(unsafe.Pointer(&data[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n), C.double(probability), C.int(trainingInt), C.int(seed),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_dropout failed (rc=%d)", rc)
	}
	return dst, nil
}

// LayerNorm
func (c *CUDAMathOps) LayerNorm(
	shape []int, eps float64, weight, bias []float64, data ...[]float64,
) ([]float64, error) {
	dModel := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dModel
	dst := make([]float64, n)
	rc := C.cuda_layernorm(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		(*C.double)(unsafe.Pointer(&bias[0])),
		C.int(numRows), C.int(dModel), C.double(eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_layernorm failed (rc=%d)", rc)
	}
	return dst, nil
}

// RMSNorm
func (c *CUDAMathOps) RMSNorm(
	shape []int, eps float64, weight []float64, data ...[]float64,
) ([]float64, error) {
	dModel := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dModel
	dst := make([]float64, n)
	rc := C.cuda_rmsnorm(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		C.int(numRows), C.int(dModel), C.double(eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_rmsnorm failed (rc=%d)", rc)
	}
	return dst, nil
}

// Sign: elementwise sign
func (c *CUDAMathOps) Sign(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.cuda_sign(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_sign failed (rc=%d)", rc)
	}
	return dst, nil
}

// Outer: outer product a[M] x b[N] → dst[M*N]
func (c *CUDAMathOps) Outer(shape []int, data ...[]float64) ([]float64, error) {
	M, N := shape[0], shape[1]
	dst := make([]float64, M*N)
	rc := C.cuda_outer(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(N),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_outer failed (rc=%d)", rc)
	}
	return dst, nil
}
