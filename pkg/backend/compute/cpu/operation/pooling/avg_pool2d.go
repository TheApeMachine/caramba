package pooling

import "math"

// AvgPool2d applies 2-D average pooling over a 4-D input [N, C, H, W].
type AvgPool2d struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
	CountIncludePad      bool
	DivisorOverride      int // 0 means use natural kernel count
}

// NewAvgPool2d creates an AvgPool2d with the given parameters.
func NewAvgPool2d(kernelH, kernelW, strideH, strideW, padH, padW, dilH, dilW int, ceil, countIncludePad bool, divisorOverride int) *AvgPool2d {
	return &AvgPool2d{
		KernelH:         kernelH,
		KernelW:         kernelW,
		StrideH:         strideH,
		StrideW:         strideW,
		PadH:            padH,
		PadW:            padW,
		DilationH:       dilH,
		DilationW:       dilW,
		CeilMode:        ceil,
		CountIncludePad: countIncludePad,
		DivisorOverride: divisorOverride,
	}
}

// Forward computes AvgPool2d.
// shape = [N, C, H, W]; data[0] = flat input of length N*C*H*W.
// Returns flat output of length N*C*H_out*W_out.
func (p *AvgPool2d) Forward(shape []int, data ...[]float64) []float64 {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	Hout := outSizeMax(H, p.KernelH, p.StrideH, p.PadH, p.DilationH, p.CeilMode)
	Wout := outSizeMax(W, p.KernelW, p.StrideW, p.PadW, p.DilationW, p.CeilMode)

	x := data[0]
	out := make([]float64, N*C*Hout*Wout)

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			baseIn := (n*C + c) * H * W
			baseOut := (n*C + c) * Hout * Wout
			for oh := 0; oh < Hout; oh++ {
				for ow := 0; ow < Wout; ow++ {
					hStart := oh*p.StrideH - p.PadH
					wStart := ow*p.StrideW - p.PadW

					var rowBuf [maxKernelElems]float64
					cnt := 0
					kernelCount := 0
					for kh := 0; kh < p.KernelH; kh++ {
						ih := hStart + kh*p.DilationH
						validH := ih >= 0 && ih < H
						for kw := 0; kw < p.KernelW; kw++ {
							kernelCount++
							iw := wStart + kw*p.DilationW
							if validH && iw >= 0 && iw < W {
								rowBuf[cnt] = x[baseIn+ih*W+iw]
								cnt++
							}
						}
					}

					divisor := cnt
					if p.DivisorOverride != 0 {
						divisor = p.DivisorOverride
					} else if p.CountIncludePad {
						divisor = kernelCount
					}

					var avg float64
					if cnt > 0 && divisor > 0 {
						avg = kernelSum(rowBuf[:cnt]) / float64(divisor)
					} else if divisor > 0 {
						avg = 0
					} else {
						avg = math.NaN()
					}
					out[baseOut+oh*Wout+ow] = avg
				}
			}
		}
	}
	return out
}
