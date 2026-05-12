package convolution

import (
	"math"
	"math/rand"
)

// Conv2d applies a 2-D convolution over input [N, InC, H, W].
//
// Weight layout: [OutC, InC/Groups, KH, KW] (row-major).
// Bias layout:   [OutC].
type Conv2d struct {
	Weight      []float64
	Bias        []float64
	InChannels  int
	OutChannels int
	KernelH     int
	KernelW     int
	StrideH     int
	StrideW     int
	PadH        int
	PadW        int
	DilationH   int
	DilationW   int
	Groups      int
}

// NewConv2d allocates a Conv2d with Kaiming-uniform weight initialisation.
func NewConv2d(inC, outC, kH, kW, strideH, strideW, padH, padW, dilH, dilW, groups int) *Conv2d {
	if strideH == 0 {
		strideH = 1
	}
	if strideW == 0 {
		strideW = 1
	}
	if dilH == 0 {
		dilH = 1
	}
	if dilW == 0 {
		dilW = 1
	}
	if groups == 0 {
		groups = 1
	}
	fanIn := (inC / groups) * kH * kW
	wBound := math.Sqrt(2.0 / float64(fanIn))
	bBound := 1.0 / math.Sqrt(float64(fanIn))

	wSize := outC * (inC / groups) * kH * kW
	w := make([]float64, wSize)
	for i := range w {
		w[i] = (rand.Float64()*2 - 1) * wBound
	}
	b := make([]float64, outC)
	for i := range b {
		b[i] = (rand.Float64()*2 - 1) * bBound
	}
	return &Conv2d{
		Weight:      w,
		Bias:        b,
		InChannels:  inC,
		OutChannels: outC,
		KernelH:     kH,
		KernelW:     kW,
		StrideH:     strideH,
		StrideW:     strideW,
		PadH:        padH,
		PadW:        padW,
		DilationH:   dilH,
		DilationW:   dilW,
		Groups:      groups,
	}
}

// Forward computes the 2-D convolution.
// shape = [N, InC, H, W]; data[0] = input.
// Returns [N, OutC, H_out, W_out].
func (c *Conv2d) Forward(shape []int, data ...[]float64) []float64 {
	n, inC, h, w := shape[0], shape[1], shape[2], shape[3]

	return conv2dForward(
		data[0], n, inC, h, w,
		c.Weight, c.Bias,
		c.OutChannels, c.KernelH, c.KernelW,
		c.StrideH, c.StrideW, c.PadH, c.PadW, c.DilationH, c.DilationW,
		c.Groups,
	)
}
