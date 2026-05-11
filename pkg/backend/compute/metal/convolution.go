//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "convolution.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// ConvolutionOps dispatches convolution kernels to the GPU via Metal.
type ConvolutionOps struct {
	metallib string
}

// NewConvolutionOps creates and initializes a ConvolutionOps.
// metallib must be the absolute path to convolution.metallib compiled from
// convolution.metal.
func NewConvolutionOps(metallib string) (*ConvolutionOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_conv_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_conv_init failed (rc=%d): check that %q exists and Metal is available", rc, metallib)
	}
	return &ConvolutionOps{metallib: metallib}, nil
}

// Forward dispatches to Conv2d with the universal signature.
// shape = [N, InC, H, W], data[0] = x, data[1] = weight (flattened), data[2] = bias.
func (m *ConvolutionOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, fmt.Errorf("metal convolution Forward: use Conv1d/Conv2d/Conv3d/ConvTranspose2d")
}

// Conv1d computes a 1-D convolution.
func (m *ConvolutionOps) Conv1d(
	x []float64,
	N, InC, L int,
	weight, bias []float64,
	OutC, K, stride, pad, dilation, groups, LOut int,
) ([]float64, error) {
	src := toFloat32(x)
	wt := toFloat32(weight)
	bs := toFloat32(bias)
	dst := make([]float32, N*OutC*LOut)

	rc := C.metal_conv1d(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(InC), C.int(L),
		C.int(OutC), C.int(K),
		C.int(stride), C.int(pad), C.int(dilation), C.int(groups),
		C.int(LOut),
		(*C.float)(unsafe.Pointer(&wt[0])),
		(*C.float)(unsafe.Pointer(&bs[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_conv1d failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// Conv2d computes a 2-D convolution.
func (m *ConvolutionOps) Conv2d(
	x []float64,
	N, InC, H, W int,
	weight, bias []float64,
	OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout int,
) ([]float64, error) {
	src := toFloat32(x)
	wt := toFloat32(weight)
	bs := toFloat32(bias)
	dst := make([]float32, N*OutC*Hout*Wout)

	rc := C.metal_conv2d(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(InC), C.int(H), C.int(W),
		C.int(OutC), C.int(KH), C.int(KW),
		C.int(sH), C.int(sW), C.int(pH), C.int(pW),
		C.int(dH), C.int(dW), C.int(groups),
		C.int(Hout), C.int(Wout),
		(*C.float)(unsafe.Pointer(&wt[0])),
		(*C.float)(unsafe.Pointer(&bs[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_conv2d failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// Conv3d computes a 3-D convolution.
func (m *ConvolutionOps) Conv3d(
	x []float64,
	N, InC, D, H, W int,
	weight, bias []float64,
	OutC, KD, KH, KW int,
	sD, sH, sW, pD, pH, pW, dD, dH, dW, groups int,
	Dout, Hout, Wout int,
) ([]float64, error) {
	src := toFloat32(x)
	wt := toFloat32(weight)
	bs := toFloat32(bias)
	dst := make([]float32, N*OutC*Dout*Hout*Wout)

	rc := C.metal_conv3d(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(InC), C.int(D), C.int(H), C.int(W),
		C.int(OutC), C.int(KD), C.int(KH), C.int(KW),
		C.int(sD), C.int(sH), C.int(sW),
		C.int(pD), C.int(pH), C.int(pW),
		C.int(dD), C.int(dH), C.int(dW),
		C.int(groups),
		C.int(Dout), C.int(Hout), C.int(Wout),
		(*C.float)(unsafe.Pointer(&wt[0])),
		(*C.float)(unsafe.Pointer(&bs[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_conv3d failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// ConvTranspose2d computes a 2-D transposed convolution.
func (m *ConvolutionOps) ConvTranspose2d(
	x []float64,
	N, InC, H, W int,
	weight, bias []float64,
	OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout int,
) ([]float64, error) {
	src := toFloat32(x)
	wt := toFloat32(weight)
	bs := toFloat32(bias)
	dst := make([]float32, N*OutC*Hout*Wout)

	rc := C.metal_conv_transpose2d(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(InC), C.int(H), C.int(W),
		C.int(OutC), C.int(KH), C.int(KW),
		C.int(sH), C.int(sW), C.int(pH), C.int(pW),
		C.int(dH), C.int(dW), C.int(groups),
		C.int(Hout), C.int(Wout),
		(*C.float)(unsafe.Pointer(&wt[0])),
		(*C.float)(unsafe.Pointer(&bs[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_conv_transpose2d failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}
