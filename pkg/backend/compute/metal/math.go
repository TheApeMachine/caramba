//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "math.h"
import "C"

import (
	"fmt"
	"unsafe"
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

func (m *MathOps) Matmul(shape []int, data ...[]float64) []float64 {
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
		return make([]float64, M*N)
	}
	return toFloat64(C_)
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

func (m *MathOps) Add(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
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
		return make([]float64, n)
	}
	return toFloat64(out)
}

// ---------------------------------------------------------------------------
// Mul
// ---------------------------------------------------------------------------

func (m *MathOps) Mul(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
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
		return make([]float64, n)
	}
	return toFloat64(out)
}

// ---------------------------------------------------------------------------
// InvSqrtDimScale — shape[-1] is the dim
// ---------------------------------------------------------------------------

func (m *MathOps) InvSqrtDimScale(shape []int, data ...[]float64) []float64 {
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
		return make([]float64, n)
	}
	return toFloat64(dst)
}

// ---------------------------------------------------------------------------
// Exp
// ---------------------------------------------------------------------------

func (m *MathOps) Exp(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	src := toFloat32(data[0])
	dst := make([]float32, n)
	rc := C.metal_exp(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return make([]float64, n)
	}
	return toFloat64(dst)
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------

func (m *MathOps) Log(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	src := toFloat32(data[0])
	dst := make([]float32, n)
	rc := C.metal_log(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return make([]float64, n)
	}
	return toFloat64(dst)
}

// ---------------------------------------------------------------------------
// Softmax — shape=[..., dim_size]
// ---------------------------------------------------------------------------

func (m *MathOps) Softmax(shape []int, data ...[]float64) []float64 {
	dimSize := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dimSize
	src := toFloat32(data[0])
	dst := make([]float32, n)
	rc := C.metal_softmax(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(numRows), C.int(dimSize),
	)
	if rc != 0 {
		return make([]float64, n)
	}
	return toFloat64(dst)
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

func (m *MathOps) LayerNorm(shape []int, eps float64, weight, bias []float64, data ...[]float64) []float64 {
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
		return make([]float64, n)
	}
	return toFloat64(dst)
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

func (m *MathOps) RMSNorm(shape []int, eps float64, weight []float64, data ...[]float64) []float64 {
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
		return make([]float64, n)
	}
	return toFloat64(dst)
}

// Sign: elementwise sign
func (m *MathOps) Sign(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	src := toFloat32(data[0])
	dst := make([]float32, n)
	rc := C.metal_sign(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return make([]float64, n)
	}
	return toFloat64(dst)
}

// Outer: outer product a[M] x b[N] → dst[M*N]
func (m *MathOps) Outer(shape []int, data ...[]float64) []float64 {
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
		return make([]float64, M*N)
	}
	return toFloat64(dst)
}
