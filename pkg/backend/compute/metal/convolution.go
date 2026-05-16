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

func validateMetalConv1d(
	input []float64,
	weight []float64,
	bias []float64,
	batch int,
	inChannels int,
	length int,
	outChannels int,
	kernelSize int,
	stride int,
	padding int,
	dilation int,
	groups int,
	lengthOut int,
) error {
	const operation = "metal.conv1d"

	if batch <= 0 || inChannels <= 0 || length <= 0 || outChannels <= 0 ||
		kernelSize <= 0 || stride <= 0 || dilation <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if inChannels%groups != 0 || outChannels%groups != 0 {
		return fmt.Errorf(
			"%s: groups=%d must divide InC=%d and OutC=%d",
			operation, groups, inChannels, outChannels,
		)
	}

	expectedInput := batch * inChannels * length

	if err := requireMetalConvLength(
		operation, "input", len(input), expectedInput,
	); err != nil {
		return err
	}

	expectedWeight := outChannels * (inChannels / groups) * kernelSize

	if err := requireMetalConvLength(
		operation, "weight", len(weight), expectedWeight,
	); err != nil {
		return err
	}

	if err := requireMetalConvLength(
		operation, "bias", len(bias), outChannels,
	); err != nil {
		return err
	}

	expectedLengthOut := (length+2*padding-dilation*(kernelSize-1)-1)/stride + 1

	if expectedLengthOut <= 0 {
		return fmt.Errorf("%s: output length=%d must be positive", operation, expectedLengthOut)
	}

	if lengthOut != expectedLengthOut {
		return fmt.Errorf(
			"%s: LOut=%d does not match expected output length %d",
			operation, lengthOut, expectedLengthOut,
		)
	}

	return nil
}

func validateMetalConv2d(
	input []float64,
	weight []float64,
	bias []float64,
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
	heightOut int,
	widthOut int,
) error {
	const operation = "metal.conv2d"

	if batch <= 0 || inChannels <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0 ||
		strideHeight <= 0 || strideWidth <= 0 ||
		dilationHeight <= 0 || dilationWidth <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if inChannels%groups != 0 || outChannels%groups != 0 {
		return fmt.Errorf(
			"%s: groups=%d must divide InC=%d and OutC=%d",
			operation, groups, inChannels, outChannels,
		)
	}

	expectedInput := batch * inChannels * height * width

	if err := requireMetalConvLength(
		operation, "input", len(input), expectedInput,
	); err != nil {
		return err
	}

	expectedWeight := outChannels * (inChannels / groups) * kernelHeight * kernelWidth

	if err := requireMetalConvLength(
		operation, "weight", len(weight), expectedWeight,
	); err != nil {
		return err
	}

	if err := requireMetalConvLength(
		operation, "bias", len(bias), outChannels,
	); err != nil {
		return err
	}

	expectedHeightOut := (height+2*padHeight-dilationHeight*(kernelHeight-1)-1)/strideHeight + 1
	expectedWidthOut := (width+2*padWidth-dilationWidth*(kernelWidth-1)-1)/strideWidth + 1

	if expectedHeightOut <= 0 || expectedWidthOut <= 0 {
		return fmt.Errorf(
			"%s: output shape [%d,%d] must be positive",
			operation, expectedHeightOut, expectedWidthOut,
		)
	}

	if heightOut != expectedHeightOut || widthOut != expectedWidthOut {
		return fmt.Errorf(
			"%s: output shape [%d,%d] does not match expected [%d,%d]",
			operation, heightOut, widthOut, expectedHeightOut, expectedWidthOut,
		)
	}

	return nil
}

func validateMetalConv3d(
	input []float64,
	weight []float64,
	bias []float64,
	batch int,
	inChannels int,
	depth int,
	height int,
	width int,
	outChannels int,
	kernelDepth int,
	kernelHeight int,
	kernelWidth int,
	strideDepth int,
	strideHeight int,
	strideWidth int,
	padDepth int,
	padHeight int,
	padWidth int,
	dilationDepth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
	depthOut int,
	heightOut int,
	widthOut int,
) error {
	const operation = "metal.conv3d"

	if batch <= 0 || inChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelDepth <= 0 || kernelHeight <= 0 ||
		kernelWidth <= 0 || strideDepth <= 0 || strideHeight <= 0 ||
		strideWidth <= 0 || dilationDepth <= 0 || dilationHeight <= 0 ||
		dilationWidth <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if inChannels%groups != 0 || outChannels%groups != 0 {
		return fmt.Errorf(
			"%s: groups=%d must divide InC=%d and OutC=%d",
			operation, groups, inChannels, outChannels,
		)
	}

	expectedInput := batch * inChannels * depth * height * width

	if err := requireMetalConvLength(
		operation, "input", len(input), expectedInput,
	); err != nil {
		return err
	}

	expectedWeight := outChannels * (inChannels / groups) * kernelDepth *
		kernelHeight * kernelWidth

	if err := requireMetalConvLength(
		operation, "weight", len(weight), expectedWeight,
	); err != nil {
		return err
	}

	if err := requireMetalConvLength(
		operation, "bias", len(bias), outChannels,
	); err != nil {
		return err
	}

	expectedDepthOut := (depth+2*padDepth-dilationDepth*(kernelDepth-1)-1)/strideDepth + 1
	expectedHeightOut := (height+2*padHeight-dilationHeight*(kernelHeight-1)-1)/strideHeight + 1
	expectedWidthOut := (width+2*padWidth-dilationWidth*(kernelWidth-1)-1)/strideWidth + 1

	if expectedDepthOut <= 0 || expectedHeightOut <= 0 || expectedWidthOut <= 0 {
		return fmt.Errorf(
			"%s: output shape [%d,%d,%d] must be positive",
			operation, expectedDepthOut, expectedHeightOut, expectedWidthOut,
		)
	}

	if depthOut != expectedDepthOut || heightOut != expectedHeightOut ||
		widthOut != expectedWidthOut {
		return fmt.Errorf(
			"%s: output shape [%d,%d,%d] does not match expected [%d,%d,%d]",
			operation, depthOut, heightOut, widthOut,
			expectedDepthOut, expectedHeightOut, expectedWidthOut,
		)
	}

	return nil
}

func validateMetalConvTranspose2d(
	input []float64,
	weight []float64,
	bias []float64,
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
	heightOut int,
	widthOut int,
) error {
	const operation = "metal.conv_transpose2d"

	if batch <= 0 || inChannels <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0 ||
		strideHeight <= 0 || strideWidth <= 0 ||
		dilationHeight <= 0 || dilationWidth <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if inChannels%groups != 0 || outChannels%groups != 0 {
		return fmt.Errorf(
			"%s: groups=%d must divide InC=%d and OutC=%d",
			operation, groups, inChannels, outChannels,
		)
	}

	expectedInput := batch * inChannels * height * width

	if err := requireMetalConvLength(
		operation, "input", len(input), expectedInput,
	); err != nil {
		return err
	}

	expectedWeight := inChannels * (outChannels / groups) * kernelHeight * kernelWidth

	if err := requireMetalConvLength(
		operation, "weight", len(weight), expectedWeight,
	); err != nil {
		return err
	}

	if err := requireMetalConvLength(
		operation, "bias", len(bias), outChannels,
	); err != nil {
		return err
	}

	expectedHeightOut := (height-1)*strideHeight - 2*padHeight +
		dilationHeight*(kernelHeight-1) + 1
	expectedWidthOut := (width-1)*strideWidth - 2*padWidth +
		dilationWidth*(kernelWidth-1) + 1

	if expectedHeightOut <= 0 || expectedWidthOut <= 0 {
		return fmt.Errorf(
			"%s: output shape [%d,%d] must be positive",
			operation, expectedHeightOut, expectedWidthOut,
		)
	}

	if heightOut != expectedHeightOut || widthOut != expectedWidthOut {
		return fmt.Errorf(
			"%s: output shape [%d,%d] does not match expected [%d,%d]",
			operation, heightOut, widthOut, expectedHeightOut, expectedWidthOut,
		)
	}

	return nil
}

func requireMetalConvLength(
	operation string,
	name string,
	actual int,
	expected int,
) error {
	if actual == expected {
		return nil
	}

	return fmt.Errorf("%s: len(%s)=%d, need %d", operation, name, actual, expected)
}
