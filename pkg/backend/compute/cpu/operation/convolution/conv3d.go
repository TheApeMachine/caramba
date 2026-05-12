package convolution

import (
	"math"
	"math/rand"
)

// Conv3d applies a 3-D convolution over input [N, InC, D, H, W].
//
// Weight layout: [OutC, InC/Groups, KD, KH, KW] (row-major).
// Bias layout:   [OutC].
type Conv3d struct {
	Weight      []float64
	Bias        []float64
	InChannels  int
	OutChannels int
	KernelD     int
	KernelH     int
	KernelW     int
	StrideD     int
	StrideH     int
	StrideW     int
	PadD        int
	PadH        int
	PadW        int
	DilationD   int
	DilationH   int
	DilationW   int
	Groups      int
}

// NewConv3d allocates a Conv3d with Kaiming-uniform weight initialisation.
func NewConv3d(inC, outC, kD, kH, kW, sD, sH, sW, pD, pH, pW, dilD, dilH, dilW, groups int) *Conv3d {
	if sD == 0 {
		sD = 1
	}
	if sH == 0 {
		sH = 1
	}
	if sW == 0 {
		sW = 1
	}
	if dilD == 0 {
		dilD = 1
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
	fanIn := (inC / groups) * kD * kH * kW
	wBound := math.Sqrt(2.0 / float64(fanIn))
	bBound := 1.0 / math.Sqrt(float64(fanIn))

	wSize := outC * (inC / groups) * kD * kH * kW
	w := make([]float64, wSize)
	for i := range w {
		w[i] = (rand.Float64()*2 - 1) * wBound
	}
	b := make([]float64, outC)
	for i := range b {
		b[i] = (rand.Float64()*2 - 1) * bBound
	}
	return &Conv3d{
		Weight:      w,
		Bias:        b,
		InChannels:  inC,
		OutChannels: outC,
		KernelD:     kD,
		KernelH:     kH,
		KernelW:     kW,
		StrideD:     sD,
		StrideH:     sH,
		StrideW:     sW,
		PadD:        pD,
		PadH:        pH,
		PadW:        pW,
		DilationD:   dilD,
		DilationH:   dilH,
		DilationW:   dilW,
		Groups:      groups,
	}
}

// Forward computes the 3-D convolution.
// shape = [N, InC, D, H, W]; data[0] = input.
// Returns [N, OutC, D_out, H_out, W_out].
func (c *Conv3d) Forward(shape []int, data ...[]float64) []float64 {
	n, inC, d, h, w := shape[0], shape[1], shape[2], shape[3], shape[4]

	return conv3dForward(
		data[0], n, inC, d, h, w,
		c.Weight, c.Bias,
		c.OutChannels, c.KernelD, c.KernelH, c.KernelW,
		c.StrideD, c.StrideH, c.StrideW,
		c.PadD, c.PadH, c.PadW,
		c.DilationD, c.DilationH, c.DilationW,
		c.Groups,
	)
}
