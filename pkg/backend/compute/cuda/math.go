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
func (c *CUDAMathOps) Matmul(shape []int, data ...[]float64) []float64 {
	M, K, N := shape[0], shape[1], shape[2]
	dst := make([]float64, M*N)
	rc := C.cuda_matmul(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(K), C.int(N),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_matmul failed (rc=%d)", rc))
	}
	return dst
}

// Add: elementwise data[0]+data[1]
func (c *CUDAMathOps) Add(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.cuda_add(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_add failed (rc=%d)", rc))
	}
	return dst
}

// Mul: elementwise data[0]*data[1]
func (c *CUDAMathOps) Mul(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.cuda_mul(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_mul failed (rc=%d)", rc))
	}
	return dst
}

// InvSqrtDimScale: data[0] * (1/sqrt(shape[-1]))
func (c *CUDAMathOps) InvSqrtDimScale(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dim := shape[len(shape)-1]
	dst := make([]float64, n)
	rc := C.cuda_inv_sqrt_dim_scale(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n), C.int(dim),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_inv_sqrt_dim_scale failed (rc=%d)", rc))
	}
	return dst
}

// Exp: elementwise exp
func (c *CUDAMathOps) Exp(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.cuda_exp(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_exp failed (rc=%d)", rc))
	}
	return dst
}

// Log: elementwise log
func (c *CUDAMathOps) Log(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.cuda_log(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_log failed (rc=%d)", rc))
	}
	return dst
}

// Softmax: shape=[...,dim_size], softmax over last dim
func (c *CUDAMathOps) Softmax(shape []int, data ...[]float64) []float64 {
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
		panic(fmt.Sprintf("cuda_softmax failed (rc=%d)", rc))
	}
	return dst
}

// LayerNorm
func (c *CUDAMathOps) LayerNorm(shape []int, eps float64, weight, bias []float64, data ...[]float64) []float64 {
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
		panic(fmt.Sprintf("cuda_layernorm failed (rc=%d)", rc))
	}
	return dst
}

// RMSNorm
func (c *CUDAMathOps) RMSNorm(shape []int, eps float64, weight []float64, data ...[]float64) []float64 {
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
		panic(fmt.Sprintf("cuda_rmsnorm failed (rc=%d)", rc))
	}
	return dst
}
