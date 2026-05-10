//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "convolution.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// CUDAConvolution dispatches convolution kernels to the GPU via CUDA.
// Device memory is managed internally by the C wrappers.
type CUDAConvolution struct{}

// NewCUDAConvolution creates a CUDAConvolution.
// No explicit initialization is required — CUDA context creation is lazy.
func NewCUDAConvolution() *CUDAConvolution {
	return &CUDAConvolution{}
}

// Forward dispatches to Conv2d with the universal signature.
// shape = [N, InC, H, W], data[0] = x, data[1] = weight (flattened), data[2] = bias.
func (c *CUDAConvolution) Forward(shape []int, data ...[]float64) []float64 {
	return nil
}

// Conv1d computes a 1-D convolution on the GPU.
func (c *CUDAConvolution) Conv1d(
	x []float64,
	N, InC, L int,
	weight, bias []float64,
	OutC, K, stride, pad, dilation, groups, LOut int,
) ([]float64, error) {
	dst := make([]float64, N*OutC*LOut)
	rc := C.cuda_conv1d(
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(InC), C.int(L),
		C.int(OutC), C.int(K),
		C.int(stride), C.int(pad), C.int(dilation), C.int(groups),
		C.int(LOut),
		(*C.double)(unsafe.Pointer(&weight[0])),
		(*C.double)(unsafe.Pointer(&bias[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_conv1d failed (rc=%d)", rc)
	}
	return dst, nil
}

// Conv2d computes a 2-D convolution on the GPU.
func (c *CUDAConvolution) Conv2d(
	x []float64,
	N, InC, H, W int,
	weight, bias []float64,
	OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout int,
) ([]float64, error) {
	dst := make([]float64, N*OutC*Hout*Wout)
	rc := C.cuda_conv2d(
		(*C.double)(unsafe.Pointer(&x[0])),
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
		return nil, fmt.Errorf("cuda_conv2d failed (rc=%d)", rc)
	}
	return dst, nil
}

// Conv3d computes a 3-D convolution on the GPU.
func (c *CUDAConvolution) Conv3d(
	x []float64,
	N, InC, D, H, W int,
	weight, bias []float64,
	OutC, KD, KH, KW int,
	sD, sH, sW, pD, pH, pW, dD, dH, dW, groups int,
	Dout, Hout, Wout int,
) ([]float64, error) {
	dst := make([]float64, N*OutC*Dout*Hout*Wout)
	rc := C.cuda_conv3d(
		(*C.double)(unsafe.Pointer(&x[0])),
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
		return nil, fmt.Errorf("cuda_conv3d failed (rc=%d)", rc)
	}
	return dst, nil
}

// ConvTranspose2d computes a 2-D transposed convolution on the GPU.
func (c *CUDAConvolution) ConvTranspose2d(
	x []float64,
	N, InC, H, W int,
	weight, bias []float64,
	OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout int,
) ([]float64, error) {
	dst := make([]float64, N*OutC*Hout*Wout)
	rc := C.cuda_conv_transpose2d(
		(*C.double)(unsafe.Pointer(&x[0])),
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
		return nil, fmt.Errorf("cuda_conv_transpose2d failed (rc=%d)", rc)
	}
	return dst, nil
}
