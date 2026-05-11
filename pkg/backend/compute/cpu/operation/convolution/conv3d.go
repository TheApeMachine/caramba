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
	x := data[0]

	// Fast path: no padding or dilation means contiguous memory access; skip the general direct nested-loop convolution fallback in applyConv3d.
	if c.DilationD == 1 && c.DilationH == 1 && c.DilationW == 1 &&
		c.PadD == 0 && c.PadH == 0 && c.PadW == 0 {
		return conv3dForwardFast(x, n, inC, d, h, w,
			c.Weight, c.Bias,
			c.OutChannels, c.KernelD, c.KernelH, c.KernelW,
			c.StrideD, c.StrideH, c.StrideW, c.Groups,
		)
	}

	kD, kH, kW := c.KernelD, c.KernelH, c.KernelW
	dOut := (d+2*c.PadD-c.DilationD*(kD-1)-1)/c.StrideD + 1
	hOut := (h+2*c.PadH-c.DilationH*(kH-1)-1)/c.StrideH + 1
	wOut := (w+2*c.PadW-c.DilationW*(kW-1)-1)/c.StrideW + 1
	outC := c.OutChannels

	out := make([]float64, n*outC*dOut*hOut*wOut)
	applyConv3d(out, x, c.Weight, c.Bias,
		n, inC, d, h, w,
		outC, kD, kH, kW,
		c.StrideD, c.StrideH, c.StrideW,
		c.PadD, c.PadH, c.PadW,
		c.DilationD, c.DilationH, c.DilationW,
		c.Groups,
		dOut, hOut, wOut)
	return out
}

// applyConv3d is the pure-Go fallback implementation.
// Platform-specific files may replace the inner dotProduct call.
func applyConv3d(
	out, x, wt, bias []float64,
	n, inC, d, h, w int,
	outC, kD, kH, kW int,
	sD, sH, sW int,
	pD, pH, pW int,
	dilD, dilH, dilW int,
	groups int,
	dOut, hOut, wOut int,
) {
	icPerGroup := inC / groups
	ocPerGroup := outC / groups

	for ni := 0; ni < n; ni++ {
		for g := 0; g < groups; g++ {
			ocStart := g * ocPerGroup
			icStart := g * icPerGroup
			for oc := ocStart; oc < ocStart+ocPerGroup; oc++ {
				kernElems := icPerGroup * kD * kH * kW
				wRow := wt[oc*kernElems : (oc+1)*kernElems]
				b := bias[oc]
				for do := 0; do < dOut; do++ {
					for ho := 0; ho < hOut; ho++ {
						for wo := 0; wo < wOut; wo++ {
							sum := b
							wIdx := 0
							for ic := 0; ic < icPerGroup; ic++ {
								absIC := icStart + ic
								for kd := 0; kd < kD; kd++ {
									di := do*sD + kd*dilD - pD
									if di < 0 || di >= d {
										wIdx += kH * kW
										continue
									}
									for kh := 0; kh < kH; kh++ {
										hi := ho*sH + kh*dilH - pH
										if hi < 0 || hi >= h {
											wIdx += kW
											continue
										}
										for kw := 0; kw < kW; kw++ {
											wi := wo*sW + kw*dilW - pW
											if wi >= 0 && wi < w {
												xIdx := ni*inC*d*h*w + absIC*d*h*w + di*h*w + hi*w + wi
												sum += x[xIdx] * wRow[wIdx]
											}
											wIdx++
										}
									}
								}
							}
							out[ni*outC*dOut*hOut*wOut+oc*dOut*hOut*wOut+do*hOut*wOut+ho*wOut+wo] = sum
						}
					}
				}
			}
		}
	}
}
