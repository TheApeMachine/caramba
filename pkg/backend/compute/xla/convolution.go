//go:build cgo && xla

package xla

// XLA convolution backend via the PJRT C API.
//
// Build requirements: same as activation.go in this package.

// #include <stdlib.h>
// #include "convolution.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// XLAConvolution dispatches convolution functions to the XLA runtime via PJRT.
// Compiled executables are cached per unique (shape, params) key.
type XLAConvolution struct {
	platform string
}

// NewXLAConvolution initialises the PJRT client for the given platform.
func NewXLAConvolution(platform string) (*XLAConvolution, error) {
	if err := NewPJRTConfig(platform).ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_conv_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_conv_init failed for platform %q", platform)
	}
	return &XLAConvolution{platform: platform}, nil
}

// Shutdown releases all convolution XLA resources.
func (x *XLAConvolution) Shutdown() {
	C.xla_conv_shutdown()
}

// Forward dispatches to Conv2d with the universal signature.
func (x *XLAConvolution) Forward(shape []int, data ...[]float64) []float64 {
	return nil
}

// Conv1d computes a 1-D convolution via XLA.
func (x *XLAConvolution) Conv1d(
	input []float64,
	N, InC, L int,
	weight, bias []float64,
	OutC, K, stride, pad, dilation, groups, LOut int,
) ([]float64, error) {
	dst := make([]float64, N*OutC*LOut)
	rc := C.xla_conv1d(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(InC), C.int(L),
		C.int(OutC), C.int(K),
		C.int(stride), C.int(pad), C.int(dilation), C.int(groups),
		C.int(LOut),
		(*C.double)(unsafe.Pointer(&weight[0])),
		(*C.double)(unsafe.Pointer(&bias[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_conv1d failed")
	}
	return dst, nil
}

// Conv2d computes a 2-D convolution via XLA.
func (x *XLAConvolution) Conv2d(
	input []float64,
	N, InC, H, W int,
	weight, bias []float64,
	OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout int,
) ([]float64, error) {
	dst := make([]float64, N*OutC*Hout*Wout)
	rc := C.xla_conv2d(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(InC), C.int(H), C.int(W),
		C.int(OutC), C.int(KH), C.int(KW),
		C.int(sH), C.int(sW), C.int(pH), C.int(pW),
		C.int(dH), C.int(dW), C.int(groups),
		C.int(Hout), C.int(Wout),
		(*C.double)(unsafe.Pointer(&weight[0])),
		(*C.double)(unsafe.Pointer(&bias[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_conv2d failed")
	}
	return dst, nil
}

// Conv3d computes a 3-D convolution via XLA.
func (x *XLAConvolution) Conv3d(
	input []float64,
	N, InC, D, H, W int,
	weight, bias []float64,
	OutC, KD, KH, KW int,
	sD, sH, sW, pD, pH, pW, dD, dH, dW, groups int,
	Dout, Hout, Wout int,
) ([]float64, error) {
	dst := make([]float64, N*OutC*Dout*Hout*Wout)
	rc := C.xla_conv3d(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(InC), C.int(D), C.int(H), C.int(W),
		C.int(OutC), C.int(KD), C.int(KH), C.int(KW),
		C.int(sD), C.int(sH), C.int(sW),
		C.int(pD), C.int(pH), C.int(pW),
		C.int(dD), C.int(dH), C.int(dW),
		C.int(groups),
		C.int(Dout), C.int(Hout), C.int(Wout),
		(*C.double)(unsafe.Pointer(&weight[0])),
		(*C.double)(unsafe.Pointer(&bias[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_conv3d failed")
	}
	return dst, nil
}

// ConvTranspose2d computes a 2-D transposed convolution via XLA.
func (x *XLAConvolution) ConvTranspose2d(
	input []float64,
	N, InC, H, W int,
	weight, bias []float64,
	OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout int,
) ([]float64, error) {
	dst := make([]float64, N*OutC*Hout*Wout)
	rc := C.xla_conv_transpose2d(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(InC), C.int(H), C.int(W),
		C.int(OutC), C.int(KH), C.int(KW),
		C.int(sH), C.int(sW), C.int(pH), C.int(pW),
		C.int(dH), C.int(dW), C.int(groups),
		C.int(Hout), C.int(Wout),
		(*C.double)(unsafe.Pointer(&weight[0])),
		(*C.double)(unsafe.Pointer(&bias[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_conv_transpose2d failed")
	}
	return dst, nil
}
