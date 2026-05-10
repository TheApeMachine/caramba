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
	x := data[0]

	kH, kW := c.KernelH, c.KernelW
	hOut := (h+2*c.PadH-c.DilationH*(kH-1)-1)/c.StrideH + 1
	wOut := (w+2*c.PadW-c.DilationW*(kW-1)-1)/c.StrideW + 1
	outC := c.OutChannels
	icPerGroup := inC / c.Groups
	ocPerGroup := outC / c.Groups

	out := make([]float64, n*outC*hOut*wOut)

	for ni := 0; ni < n; ni++ {
		for g := 0; g < c.Groups; g++ {
			ocStart := g * ocPerGroup
			icStart := g * icPerGroup
			for oc := ocStart; oc < ocStart+ocPerGroup; oc++ {
				wKernelSize := icPerGroup * kH * kW
				wRow := c.Weight[oc*wKernelSize : (oc+1)*wKernelSize]
				bias := c.Bias[oc]
				for ho := 0; ho < hOut; ho++ {
					for wo := 0; wo < wOut; wo++ {
						sum := bias
						wIdx := 0
						for ic := 0; ic < icPerGroup; ic++ {
							absIC := icStart + ic
							for kh := 0; kh < kH; kh++ {
								hi := ho*c.StrideH + kh*c.DilationH - c.PadH
								if hi < 0 || hi >= h {
									wIdx += kW
									continue
								}
								for kw := 0; kw < kW; kw++ {
									wi := wo*c.StrideW + kw*c.DilationW - c.PadW
									if wi >= 0 && wi < w {
										sum += x[ni*inC*h*w+absIC*h*w+hi*w+wi] * wRow[wIdx]
									}
									wIdx++
								}
							}
						}
						out[ni*outC*hOut*wOut+oc*hOut*wOut+ho*wOut+wo] = sum
					}
				}
			}
		}
	}
	return out
}
