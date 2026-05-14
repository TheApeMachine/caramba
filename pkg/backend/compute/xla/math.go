//go:build cgo && xla

package xla

// XLA math backend via the PJRT C API.
//
// Configure PJRT paths under compute.xla in cmd/asset/config.yml before runtime validation.

// #include <stdlib.h>
// #include "xla_math.h"
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
	config, err := NewPJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	if err := config.ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_math_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_math_init failed for platform %q", platform)
	}
	return &XLAMathOps{platform: platform}, nil
}

// Shutdown releases all PJRT math resources.
func (x *XLAMathOps) Shutdown() { C.xla_math_shutdown() }

func (x *XLAMathOps) Matmul(shape []int, data ...[]float64) ([]float64, error) {
	M, K, N := shape[0], shape[1], shape[2]
	dst := make([]float64, M*N)
	rc := C.xla_matmul(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(K), C.int(N),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_matmul failed: rc=%d", rc)
	}
	return dst, nil
}

func (x *XLAMathOps) Add(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.xla_add(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_add failed")
	}
	return dst, nil
}

func (x *XLAMathOps) Mul(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.xla_mul(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_mul failed")
	}
	return dst, nil
}

func (x *XLAMathOps) InvSqrtDimScale(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dim := shape[len(shape)-1]
	dst := make([]float64, n)
	rc := C.xla_inv_sqrt_dim_scale(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n), C.int(dim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_inv_sqrt_dim_scale failed")
	}
	return dst, nil
}

func (x *XLAMathOps) Exp(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.xla_exp(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_exp failed")
	}
	return dst, nil
}

func (x *XLAMathOps) Log(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.xla_log(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_log failed")
	}
	return dst, nil
}

func (x *XLAMathOps) Softmax(shape []int, data ...[]float64) ([]float64, error) {
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
		return nil, fmt.Errorf("xla_softmax failed")
	}
	return dst, nil
}

func (x *XLAMathOps) LogSumExp(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("xla_logsumexp: shape is required")
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("xla_logsumexp: input[0] is required")
	}

	dimSize := shape[len(shape)-1]
	n := len(data[0])

	if n == 0 {
		return []float64{}, nil
	}

	if dimSize <= 0 || n%dimSize != 0 {
		return nil, fmt.Errorf("xla_logsumexp: invalid dim size %d for length %d", dimSize, n)
	}

	numRows := n / dimSize
	dst := make([]float64, numRows)
	rc := C.xla_logsumexp(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(numRows), C.int(dimSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_logsumexp failed: rc=%d", rc)
	}
	return dst, nil
}

func (x *XLAMathOps) Dropout(
	probability float64,
	training bool,
	seed int,
	data []float64,
) ([]float64, error) {
	if probability < 0 || probability >= 1 {
		return nil, fmt.Errorf("xla_dropout: probability must be in [0,1)")
	}

	if len(data) == 0 {
		return []float64{}, nil
	}

	trainingFlag := 0
	if training {
		trainingFlag = 1
	}

	dst := make([]float64, len(data))
	rc := C.xla_dropout(
		(*C.double)(unsafe.Pointer(&data[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(len(data)),
		C.double(probability),
		C.int(trainingFlag),
		C.int(seed),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_dropout failed: rc=%d", rc)
	}
	return dst, nil
}

func (x *XLAMathOps) LayerNorm(
	shape []int, eps float64, weight, bias []float64, data ...[]float64,
) ([]float64, error) {
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
		return nil, fmt.Errorf("xla_layernorm failed")
	}
	return dst, nil
}

func (x *XLAMathOps) RMSNorm(
	shape []int, eps float64, weight []float64, data ...[]float64,
) ([]float64, error) {
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
		return nil, fmt.Errorf("xla_rmsnorm failed")
	}
	return dst, nil
}

func (x *XLAMathOps) Sign(shape []int, data ...[]float64) ([]float64, error) {
	n := len(data[0])
	dst := make([]float64, n)
	rc := C.xla_sign(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_sign failed")
	}
	return dst, nil
}

func (x *XLAMathOps) Outer(shape []int, data ...[]float64) ([]float64, error) {
	M, N := shape[0], shape[1]
	dst := make([]float64, M*N)
	rc := C.xla_outer(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(N),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_outer failed")
	}
	return dst, nil
}
