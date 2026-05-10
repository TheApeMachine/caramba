//go:build cgo && xla

package xla

// XLA pooling backend via the PJRT C API.
//
// Build requirements: same as activation.go.
// Pooling modules are compiled on-demand and cached per parameter set.

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ldl -lstdc++
// #include "pooling.h"
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// XLAPooling dispatches 2-D pooling operations to the XLA runtime via PJRT.
type XLAPooling struct {
	platform string
}

// NewXLAPooling initialises the PJRT client for pooling.
// Call after (or instead of) NewXLAActivation — they share the same PJRT client.
func NewXLAPooling(platform string) (*XLAPooling, error) {
	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_pooling_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_pooling_init failed for platform %q", platform)
	}
	return &XLAPooling{platform: platform}, nil
}

// Shutdown releases pooling PJRT resources.
func (x *XLAPooling) Shutdown() {
	C.xla_pooling_shutdown()
}

// ---------------------------------------------------------------------------
// outSize helper
// ---------------------------------------------------------------------------

func xlaOutSize(in, kernel, stride, pad, dilation int, ceil bool) int {
	eff := dilation*(kernel-1) + 1
	if ceil {
		return int(math.Ceil(float64(in+2*pad-eff)/float64(stride))) + 1
	}
	return (in+2*pad-eff)/stride + 1
}

// ---------------------------------------------------------------------------
// MaxPool2d
// ---------------------------------------------------------------------------

// XLAMaxPool2dParams carries hyperparameters for max pooling.
type XLAMaxPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
}

// MaxPool2d computes 2-D max pooling via XLA.
func (x *XLAPooling) MaxPool2d(shape []int, params XLAMaxPool2dParams, data []float64) ([]float64, error) {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	Hout := xlaOutSize(H, params.KernelH, params.StrideH, params.PadH, params.DilationH, params.CeilMode)
	Wout := xlaOutSize(W, params.KernelW, params.StrideW, params.PadW, params.DilationW, params.CeilMode)
	dst := make([]float64, N*C*Hout*Wout)

	rc := C.xla_max_pool2d(
		(*C.double)(unsafe.Pointer(&data[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(C), C.int(H), C.int(W),
		C.int(params.KernelH), C.int(params.KernelW),
		C.int(params.StrideH), C.int(params.StrideW),
		C.int(params.PadH), C.int(params.PadW),
		C.int(params.DilationH), C.int(params.DilationW),
		C.int(Hout), C.int(Wout),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_max_pool2d failed")
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// AvgPool2d
// ---------------------------------------------------------------------------

// XLAAvgPool2dParams carries hyperparameters for average pooling.
type XLAAvgPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
	CountIncludePad      bool
	DivisorOverride      int
}

// AvgPool2d computes 2-D average pooling via XLA.
func (x *XLAPooling) AvgPool2d(shape []int, params XLAAvgPool2dParams, data []float64) ([]float64, error) {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	Hout := xlaOutSize(H, params.KernelH, params.StrideH, params.PadH, params.DilationH, params.CeilMode)
	Wout := xlaOutSize(W, params.KernelW, params.StrideW, params.PadW, params.DilationW, params.CeilMode)
	dst := make([]float64, N*C*Hout*Wout)

	cip := 0
	if params.CountIncludePad {
		cip = 1
	}

	rc := C.xla_avg_pool2d(
		(*C.double)(unsafe.Pointer(&data[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(C), C.int(H), C.int(W),
		C.int(params.KernelH), C.int(params.KernelW),
		C.int(params.StrideH), C.int(params.StrideW),
		C.int(params.PadH), C.int(params.PadW),
		C.int(params.DilationH), C.int(params.DilationW),
		C.int(Hout), C.int(Wout),
		C.int(cip), C.int(params.DivisorOverride),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_avg_pool2d failed")
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2d
// ---------------------------------------------------------------------------

// AdaptiveAvgPool2d computes adaptive average pooling to [OutH, OutW] via XLA.
func (x *XLAPooling) AdaptiveAvgPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	dst := make([]float64, N*C*outH*outW)

	rc := C.xla_adaptive_avg_pool2d(
		(*C.double)(unsafe.Pointer(&data[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(C), C.int(H), C.int(W),
		C.int(outH), C.int(outW),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_adaptive_avg_pool2d failed")
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// AdaptiveMaxPool2d
// ---------------------------------------------------------------------------

// AdaptiveMaxPool2d computes adaptive max pooling to [OutH, OutW] via XLA.
func (x *XLAPooling) AdaptiveMaxPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	dst := make([]float64, N*C*outH*outW)

	rc := C.xla_adaptive_max_pool2d(
		(*C.double)(unsafe.Pointer(&data[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(C), C.int(H), C.int(W),
		C.int(outH), C.int(outW),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_adaptive_max_pool2d failed")
	}
	return dst, nil
}

// Forward implements the universal operation interface using MaxPool2d.
func (x *XLAPooling) Forward(shape []int, data ...[]float64) []float64 {
	p := XLAMaxPool2dParams{
		KernelH: 3, KernelW: 3,
		StrideH: 1, StrideW: 1,
		PadH: 0, PadW: 0,
		DilationH: 1, DilationW: 1,
	}
	out, _ := x.MaxPool2d(shape, p, data[0])
	return out
}
