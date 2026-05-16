//go:build cgo && xla

package xla

// XLA pooling backend via the PJRT C API.
//
// Build requirements: same as activation.go.
// Pooling modules are compiled on-demand and cached per parameter set.

// #include <stdlib.h>
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
	config, err := newRuntimePJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	cp := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_pooling_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_pooling_init failed for platform %q", config.Platform)
	}

	return &XLAPooling{platform: config.Platform}, nil
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

func validateXLAPoolingInput(method string, shape []int, data []float64) (int, int, int, int, error) {
	if len(shape) < 4 {
		return 0, 0, 0, 0, fmt.Errorf("xla pooling %s: shape rank must be >= 4", method)
	}

	if len(data) == 0 {
		return 0, 0, 0, 0, fmt.Errorf("xla pooling %s: data slice is empty", method)
	}

	N, C, H, W := shape[0], shape[1], shape[2], shape[3]

	if N <= 0 || C <= 0 || H <= 0 || W <= 0 {
		return 0, 0, 0, 0, fmt.Errorf(
			"xla pooling %s: shape dimensions must be positive, got N=%d C=%d H=%d W=%d",
			method, N, C, H, W,
		)
	}

	expectedSize := N * C * H * W

	if len(data) != expectedSize {
		return 0, 0, 0, 0, fmt.Errorf(
			"xla pooling %s: data length %d != expected %d",
			method, len(data), expectedSize,
		)
	}

	return N, C, H, W, nil
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
	N, C, H, W, err := validateXLAPoolingInput("MaxPool2d", shape, data)
	if err != nil {
		return nil, err
	}

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
	N, C, H, W, err := validateXLAPoolingInput("AvgPool2d", shape, data)
	if err != nil {
		return nil, err
	}

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
	N, C, H, W, err := validateXLAPoolingInput("AdaptiveAvgPool2d", shape, data)
	if err != nil {
		return nil, err
	}

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
	N, C, H, W, err := validateXLAPoolingInput("AdaptiveMaxPool2d", shape, data)
	if err != nil {
		return nil, err
	}

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

// Forward dispatches MaxPool2d.
func (x *XLAPooling) Forward(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return []float64{}, nil
	}

	if len(shape) < 4 {
		return nil, fmt.Errorf("xla pooling Forward: shape rank must be >= 4")
	}

	p := XLAMaxPool2dParams{
		KernelH: 3, KernelW: 3,
		StrideH: 1, StrideW: 1,
		PadH: 0, PadW: 0,
		DilationH: 1, DilationW: 1,
	}

	return x.MaxPool2d(shape, p, data[0])
}
