//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "convolution.h"
import "C"

import (
	"fmt"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
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
	if err := validateMetalConv1d(
		x, weight, bias, N, InC, L, OutC, K, stride, pad, dilation,
		groups, LOut,
	); err != nil {
		return nil, err
	}

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
	if err := validateMetalConv2d(
		x, weight, bias, N, InC, H, W, OutC, KH, KW, sH, sW, pH,
		pW, dH, dW, groups, Hout, Wout,
	); err != nil {
		return nil, err
	}

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

func (m *ConvolutionOps) Conv2dTensor(
	input computetensor.Float64Tensor,
	weight computetensor.Float64Tensor,
	bias computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	batch int,
	inChannels int,
	height int,
	width int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
	strideHeight int,
	strideWidth int,
	padHeight int,
	padWidth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	metalWeight, err := requireMetalTensor(weight)

	if err != nil {
		return nil, err
	}

	metalBias, err := requireMetalTensor(bias)

	if err != nil {
		return nil, err
	}

	outputDims := outputShape.Dims()

	if len(outputDims) != 4 {
		return nil, fmt.Errorf("metal.conv2d: output shape must be NCHW rank 4")
	}

	if err := validateMetalConv2dLengths(
		metalInput.Len(),
		metalWeight.Len(),
		metalBias.Len(),
		batch,
		inChannels,
		height,
		width,
		outChannels,
		kernelHeight,
		kernelWidth,
		strideHeight,
		strideWidth,
		padHeight,
		padWidth,
		dilationHeight,
		dilationWidth,
		groups,
		outputDims[2],
		outputDims[3],
	); err != nil {
		return nil, err
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_conv2d_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(batch),
		C.int(inChannels),
		C.int(height),
		C.int(width),
		C.int(outChannels),
		C.int(kernelHeight),
		C.int(kernelWidth),
		C.int(strideHeight),
		C.int(strideWidth),
		C.int(padHeight),
		C.int(padWidth),
		C.int(dilationHeight),
		C.int(dilationWidth),
		C.int(groups),
		C.int(outputDims[2]),
		C.int(outputDims[3]),
		metalWeight.buffer,
		metalBias.buffer,
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_conv2d_tensor failed (rc=%d)", rc)
	}

	return output, nil
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
	if err := validateMetalConv3d(
		x, weight, bias, N, InC, D, H, W, OutC, KD, KH, KW, sD,
		sH, sW, pD, pH, pW, dD, dH, dW, groups, Dout, Hout, Wout,
	); err != nil {
		return nil, err
	}

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

func (m *ConvolutionOps) ConvTranspose2dTensor(
	input computetensor.Float64Tensor,
	weight computetensor.Float64Tensor,
	bias computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	batch int,
	inChannels int,
	height int,
	width int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
	strideHeight int,
	strideWidth int,
	padHeight int,
	padWidth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
	outPadHeight int,
	outPadWidth int,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	metalWeight, err := requireMetalTensor(weight)

	if err != nil {
		return nil, err
	}

	metalBias, err := requireMetalTensor(bias)

	if err != nil {
		return nil, err
	}

	outputDims := outputShape.Dims()

	if len(outputDims) != 4 {
		return nil, fmt.Errorf("metal.conv_transpose2d: output shape must be NCHW rank 4")
	}

	if err := validateMetalConvTranspose2dLengths(
		metalInput.Len(),
		metalWeight.Len(),
		metalBias.Len(),
		batch,
		inChannels,
		height,
		width,
		outChannels,
		kernelHeight,
		kernelWidth,
		strideHeight,
		strideWidth,
		padHeight,
		padWidth,
		dilationHeight,
		dilationWidth,
		groups,
		outPadHeight,
		outPadWidth,
		outputDims[2],
		outputDims[3],
	); err != nil {
		return nil, err
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_conv_transpose2d_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(batch),
		C.int(inChannels),
		C.int(height),
		C.int(width),
		C.int(outChannels),
		C.int(kernelHeight),
		C.int(kernelWidth),
		C.int(strideHeight),
		C.int(strideWidth),
		C.int(padHeight),
		C.int(padWidth),
		C.int(dilationHeight),
		C.int(dilationWidth),
		C.int(groups),
		C.int(outputDims[2]),
		C.int(outputDims[3]),
		metalWeight.buffer,
		metalBias.buffer,
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_conv_transpose2d_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

// ConvTranspose2d computes a 2-D transposed convolution.
func (m *ConvolutionOps) ConvTranspose2d(
	x []float64,
	N, InC, H, W int,
	weight, bias []float64,
	OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout int,
) ([]float64, error) {
	if err := validateMetalConvTranspose2d(
		x, weight, bias, N, InC, H, W, OutC, KH, KW, sH, sW, pH,
		pW, dH, dW, groups, Hout, Wout,
	); err != nil {
		return nil, err
	}

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
