//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "pooling.h"
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// PoolingOps dispatches 2-D pooling kernels to the GPU via Metal.
// All kernels operate on float32 internally; float64 inputs/outputs are
// converted on the host.
type PoolingOps struct {
	metallib string
}

// NewPoolingOps creates and initialises a PoolingOps.
// metallib must be the absolute path to pooling.metallib compiled from
// pooling.metal via:
//
//	xcrun -sdk macosx metal -c pooling.metal -o pooling.air
//	xcrun -sdk macosx metallib pooling.air -o pooling.metallib
func NewPoolingOps(metallib string) (*PoolingOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_pooling_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_pooling_init failed (rc=%d): check that %q exists", rc, metallib)
	}
	return &PoolingOps{metallib: metallib}, nil
}

// ---------------------------------------------------------------------------
// outSize helper — mirrors the CPU formula.
// ---------------------------------------------------------------------------

func metalOutSize(in, kernel, stride, pad, dilation int, ceil bool) int {
	eff := dilation*(kernel-1) + 1
	if ceil {
		return int(math.Ceil(float64(in+2*pad-eff)/float64(stride))) + 1
	}
	return (in+2*pad-eff)/stride + 1
}

// ---------------------------------------------------------------------------
// MaxPool2d
// shape = [N, Ch, H, W]; data[0] = input.
// ---------------------------------------------------------------------------

// MaxPool2dParams carries pooling hyperparameters.
type MaxPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
}

// MaxPool2d computes 2-D max pooling on the GPU.
func (m *PoolingOps) MaxPool2d(shape []int, params MaxPool2dParams, data []float64) ([]float64, error) {
	N, Ch, H, W := shape[0], shape[1], shape[2], shape[3]
	Hout := metalOutSize(H, params.KernelH, params.StrideH, params.PadH, params.DilationH, params.CeilMode)
	Wout := metalOutSize(W, params.KernelW, params.StrideW, params.PadW, params.DilationW, params.CeilMode)

	src := toFloat32(data)
	dst := make([]float32, N*Ch*Hout*Wout)

	rc := C.metal_max_pool2d(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(Ch), C.int(H), C.int(W),
		C.int(params.KernelH), C.int(params.KernelW),
		C.int(params.StrideH), C.int(params.StrideW),
		C.int(params.PadH), C.int(params.PadW),
		C.int(params.DilationH), C.int(params.DilationW),
		C.int(Hout), C.int(Wout),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_max_pool2d failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ---------------------------------------------------------------------------
// AvgPool2d
// ---------------------------------------------------------------------------

// AvgPool2dParams carries pooling hyperparameters for average pooling.
type AvgPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
	CountIncludePad      bool
	DivisorOverride      int
}

// AvgPool2d computes 2-D average pooling on the GPU.
func (m *PoolingOps) AvgPool2d(shape []int, params AvgPool2dParams, data []float64) ([]float64, error) {
	N, Ch, H, W := shape[0], shape[1], shape[2], shape[3]
	Hout := metalOutSize(H, params.KernelH, params.StrideH, params.PadH, params.DilationH, params.CeilMode)
	Wout := metalOutSize(W, params.KernelW, params.StrideW, params.PadW, params.DilationW, params.CeilMode)

	src := toFloat32(data)
	dst := make([]float32, N*Ch*Hout*Wout)

	cip := 0
	if params.CountIncludePad {
		cip = 1
	}

	rc := C.metal_avg_pool2d(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(Ch), C.int(H), C.int(W),
		C.int(params.KernelH), C.int(params.KernelW),
		C.int(params.StrideH), C.int(params.StrideW),
		C.int(params.PadH), C.int(params.PadW),
		C.int(params.DilationH), C.int(params.DilationW),
		C.int(Hout), C.int(Wout),
		C.int(cip), C.int(params.DivisorOverride),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_avg_pool2d failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2d
// ---------------------------------------------------------------------------

// AdaptiveAvgPool2d computes adaptive average pooling to [OutH, OutW] on the GPU.
func (m *PoolingOps) AdaptiveAvgPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	N, Ch, H, W := shape[0], shape[1], shape[2], shape[3]
	src := toFloat32(data)
	dst := make([]float32, N*Ch*outH*outW)

	rc := C.metal_adaptive_avg_pool2d(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(Ch), C.int(H), C.int(W),
		C.int(outH), C.int(outW),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_adaptive_avg_pool2d failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ---------------------------------------------------------------------------
// AdaptiveMaxPool2d
// ---------------------------------------------------------------------------

// AdaptiveMaxPool2d computes adaptive max pooling to [OutH, OutW] on the GPU.
func (m *PoolingOps) AdaptiveMaxPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	N, Ch, H, W := shape[0], shape[1], shape[2], shape[3]
	src := toFloat32(data)
	dst := make([]float32, N*Ch*outH*outW)

	rc := C.metal_adaptive_max_pool2d(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(Ch), C.int(H), C.int(W),
		C.int(outH), C.int(outW),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_adaptive_max_pool2d failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ---------------------------------------------------------------------------
// Forward — universal signature; dispatches MaxPool2d with default 3×3 k=1 params.
// ---------------------------------------------------------------------------

// Forward implements the universal operation interface using MaxPool2d.
// shape=[N,Ch,H,W]; data[0]=input.
func (m *PoolingOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("metal pooling Forward: missing input data[0]")
	}

	p := MaxPool2dParams{
		KernelH: 3, KernelW: 3,
		StrideH: 1, StrideW: 1,
		PadH: 0, PadW: 0,
		DilationH: 1, DilationW: 1,
	}

	return m.MaxPool2d(shape, p, data[0])
}
