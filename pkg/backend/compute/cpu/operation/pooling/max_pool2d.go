package pooling

import "math"

// MaxPool2d applies 2-D max pooling over a 4-D input [N, C, H, W].
type MaxPool2d struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
}

// NewMaxPool2d creates a MaxPool2d with the given parameters.
func NewMaxPool2d(kernelH, kernelW, strideH, strideW, padH, padW, dilH, dilW int, ceil bool) *MaxPool2d {
	return &MaxPool2d{
		KernelH:   kernelH,
		KernelW:   kernelW,
		StrideH:   strideH,
		StrideW:   strideW,
		PadH:      padH,
		PadW:      padW,
		DilationH: dilH,
		DilationW: dilW,
		CeilMode:  ceil,
	}
}

// outSize computes the output spatial dimension.
func outSizeMax(in, kernel, stride, pad, dilation int, ceil bool) int {
	eff := dilation*(kernel-1) + 1
	if ceil {
		return int(math.Ceil(float64(in+2*pad-eff)/float64(stride))) + 1
	}
	return (in+2*pad-eff)/stride + 1
}

// Forward computes MaxPool2d.
// shape = [N, C, H, W]; data[0] = flat input of length N*C*H*W.
// Returns flat output of length N*C*H_out*W_out.
func (p *MaxPool2d) Forward(shape []int, data ...[]float64) []float64 {
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

					// Collect kernel values into a scratch slice for SIMD reduce.
					var rowBuf [maxKernelElems]float64
					cnt := 0
					for kh := 0; kh < p.KernelH; kh++ {
						ih := hStart + kh*p.DilationH
						if ih < 0 || ih >= H {
							continue
						}
						rowStart := baseIn + ih*W
						for kw := 0; kw < p.KernelW; kw++ {
							iw := wStart + kw*p.DilationW
							if iw < 0 || iw >= W {
								continue
							}
							rowBuf[cnt] = x[rowStart+iw]
							cnt++
						}
					}
					var maxVal float64
					if cnt == 0 {
						maxVal = math.Inf(-1)
					} else {
						maxVal = kernelMax(rowBuf[:cnt])
					}
					out[baseOut+oh*Wout+ow] = maxVal
				}
			}
		}
	}
	return out
}

// maxKernelElems is a generous upper bound for stack-allocated kernel buffers.
const maxKernelElems = 1024
