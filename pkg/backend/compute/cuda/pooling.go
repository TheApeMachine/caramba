//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "pooling.h"
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// CUDAPooling dispatches 2-D pooling kernels to the GPU via CUDA.
// All kernels operate on double (float64) directly.
type CUDAPooling struct{}

// NewCUDAPooling creates a CUDAPooling.
func NewCUDAPooling() *CUDAPooling {
	return &CUDAPooling{}
}

// ---------------------------------------------------------------------------
// outSize helper
// ---------------------------------------------------------------------------

func cudaOutSize(in, kernel, stride, pad, dilation int, ceil bool) int {
	eff := dilation*(kernel-1) + 1
	if ceil {
		return int(math.Ceil(float64(in+2*pad-eff)/float64(stride))) + 1
	}
	return (in+2*pad-eff)/stride + 1
}

// ---------------------------------------------------------------------------
// MaxPool2d
// ---------------------------------------------------------------------------

// CUDAMaxPool2dParams carries hyperparameters for max pooling.
type CUDAMaxPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
}

// MaxPool2d computes 2-D max pooling on the CUDA device.
// shape=[N,C,H,W]; returns output of length N*C*Hout*Wout.
func (c *CUDAPooling) MaxPool2d(shape []int, params CUDAMaxPool2dParams, data []float64) ([]float64, error) {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	Hout := cudaOutSize(H, params.KernelH, params.StrideH, params.PadH, params.DilationH, params.CeilMode)
	Wout := cudaOutSize(W, params.KernelW, params.StrideW, params.PadW, params.DilationW, params.CeilMode)
	dst := make([]float64, N*C*Hout*Wout)

	rc := C.cuda_max_pool2d(
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
		return nil, fmt.Errorf("cuda_max_pool2d failed (rc=%d)", rc)
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// AvgPool2d
// ---------------------------------------------------------------------------

// CUDAAvgPool2dParams carries hyperparameters for average pooling.
type CUDAAvgPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
	CountIncludePad      bool
	DivisorOverride      int
}

// AvgPool2d computes 2-D average pooling on the CUDA device.
func (c *CUDAPooling) AvgPool2d(shape []int, params CUDAAvgPool2dParams, data []float64) ([]float64, error) {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	Hout := cudaOutSize(H, params.KernelH, params.StrideH, params.PadH, params.DilationH, params.CeilMode)
	Wout := cudaOutSize(W, params.KernelW, params.StrideW, params.PadW, params.DilationW, params.CeilMode)
	dst := make([]float64, N*C*Hout*Wout)

	cip := 0
	if params.CountIncludePad {
		cip = 1
	}

	rc := C.cuda_avg_pool2d(
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
		return nil, fmt.Errorf("cuda_avg_pool2d failed (rc=%d)", rc)
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2d
// ---------------------------------------------------------------------------

// AdaptiveAvgPool2d computes adaptive average pooling to [OutH, OutW].
func (c *CUDAPooling) AdaptiveAvgPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	dst := make([]float64, N*C*outH*outW)

	rc := C.cuda_adaptive_avg_pool2d(
		(*C.double)(unsafe.Pointer(&data[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(C), C.int(H), C.int(W),
		C.int(outH), C.int(outW),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_adaptive_avg_pool2d failed (rc=%d)", rc)
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// AdaptiveMaxPool2d
// ---------------------------------------------------------------------------

// AdaptiveMaxPool2d computes adaptive max pooling to [OutH, OutW].
func (c *CUDAPooling) AdaptiveMaxPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	dst := make([]float64, N*C*outH*outW)

	rc := C.cuda_adaptive_max_pool2d(
		(*C.double)(unsafe.Pointer(&data[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(N), C.int(C), C.int(H), C.int(W),
		C.int(outH), C.int(outW),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_adaptive_max_pool2d failed (rc=%d)", rc)
	}
	return dst, nil
}

// Forward implements the universal operation interface using MaxPool2d.
// shape=[N,C,H,W]; data[0]=input.
func (c *CUDAPooling) Forward(shape []int, data ...[]float64) []float64 {
	p := CUDAMaxPool2dParams{
		KernelH: 3, KernelW: 3,
		StrideH: 1, StrideW: 1,
		PadH: 0, PadW: 0,
		DilationH: 1, DilationW: 1,
	}
	out, _ := c.MaxPool2d(shape, p, data[0])
	return out
}
