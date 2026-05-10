//go:build cgo && xla

package xla

// XLA math backend via the PJRT C API.
//
// Build requirements:
//   - XLA headers on the include path (set XLA_INCLUDE via CGO_CPPFLAGS)
//   - PJRT plugin shared library on LD_LIBRARY_PATH
//
// Example build:
//   CGO_CPPFLAGS="-I/path/to/xla" \
//   CGO_LDFLAGS="-ldl -lstdc++" \
//   go build -tags "cgo xla" ./backend/compute/xla/

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ldl -lstdc++
// #include <stdlib.h>
// #include "math.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// XLAMathOps dispatches math operations to the XLA runtime via PJRT.
type XLAMathOps struct {
	platform string
}

// NewMathOps initialises the PJRT client for the given platform ("cpu"/"gpu").
func NewMathOps(platform string) (*XLAMathOps, error) {
	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))
	if rc := C.xla_math_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_math_init failed for platform %q", platform)
	}
	return &XLAMathOps{platform: platform}, nil
}

// Shutdown releases all PJRT math resources.
func (x *XLAMathOps) Shutdown() { C.xla_math_shutdown() }

func (x *XLAMathOps) Matmul(shape []int, data ...[]float64) []float64 {
	M, K, N := shape[0], shape[1], shape[2]
	dst := make([]float64, M*N)
	rc := C.xla_matmul(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(K), C.int(N),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_matmul failed"))
	}
	return dst
}

func (x *XLAMathOps) Add(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.xla_add(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_add failed"))
	}
	return dst
}

func (x *XLAMathOps) Mul(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.xla_mul(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_mul failed"))
	}
	return dst
}

func (x *XLAMathOps) InvSqrtDimScale(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dim := shape[len(shape)-1]
	dst := make([]float64, n)
	rc := C.xla_inv_sqrt_dim_scale(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n), C.int(dim),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_inv_sqrt_dim_scale failed"))
	}
	return dst
}

func (x *XLAMathOps) Exp(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.xla_exp(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_exp failed"))
	}
	return dst
}

func (x *XLAMathOps) Log(shape []int, data ...[]float64) []float64 {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.xla_log(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_log failed"))
	}
	return dst
}

func (x *XLAMathOps) Softmax(shape []int, data ...[]float64) []float64 {
	dimSize := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dimSize
	dst := make([]float64, n)
	rc := C.xla_softmax(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(numRows), C.int(dimSize),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_softmax failed"))
	}
	return dst
}

func (x *XLAMathOps) LayerNorm(shape []int, eps float64, weight, bias []float64, data ...[]float64) []float64 {
	dModel := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dModel
	dst := make([]float64, n)
	rc := C.xla_layernorm(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		(*C.double)(unsafe.Pointer(&bias[0])),
		C.int(numRows), C.int(dModel), C.double(eps),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_layernorm failed"))
	}
	return dst
}

func (x *XLAMathOps) RMSNorm(shape []int, eps float64, weight []float64, data ...[]float64) []float64 {
	dModel := shape[len(shape)-1]
	n := len(data[0])
	numRows := n / dModel
	dst := make([]float64, n)
	rc := C.xla_rmsnorm(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		C.int(numRows), C.int(dModel), C.double(eps),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_rmsnorm failed"))
	}
	return dst
}
