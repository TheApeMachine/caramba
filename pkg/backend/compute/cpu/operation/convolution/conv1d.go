package convolution

import (
	"math"
	"math/rand"
)

// Conv1d applies a 1-D convolution over an input of shape [N, InC, L].
//
// Weight layout: [OutC, InC/Groups, K] (row-major, K is the innermost dim).
// Bias layout:   [OutC].
type Conv1d struct {
	Weight     []float64
	Bias       []float64
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Dilation    int
	Groups      int
}

// NewConv1d allocates a Conv1d with Kaiming-uniform weight initialisation.
func NewConv1d(inC, outC, kernelSize, stride, padding, dilation, groups int) *Conv1d {
	if stride == 0 {
		stride = 1
	}
	if dilation == 0 {
		dilation = 1
	}
	if groups == 0 {
		groups = 1
	}
	fanIn := (inC / groups) * kernelSize
	wBound := math.Sqrt(2.0 / float64(fanIn))
	bBound := 1.0 / math.Sqrt(float64(fanIn))

	wSize := outC * (inC / groups) * kernelSize
	w := make([]float64, wSize)
	for i := range w {
		w[i] = (rand.Float64()*2 - 1) * wBound
	}
	b := make([]float64, outC)
	for i := range b {
		b[i] = (rand.Float64()*2 - 1) * bBound
	}
	return &Conv1d{
		Weight:      w,
		Bias:        b,
		InChannels:  inC,
		OutChannels: outC,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		Dilation:    dilation,
		Groups:      groups,
	}
}

// Forward computes the 1-D convolution.
// shape = [N, InC, L]; data[0] = input flattened in that order.
// Returns output flattened as [N, OutC, L_out].
func (c *Conv1d) Forward(shape []int, data ...[]float64) []float64 {
	n, inC, l := shape[0], shape[1], shape[2]
	x := data[0]

	if c.Dilation == 1 && c.Padding == 0 {
		return conv1dForwardFast(x, n, inC, l,
			c.Weight, c.Bias,
			c.OutChannels, c.KernelSize, c.Stride, c.Groups,
		)
	}

	k := c.KernelSize
	lOut := (l+2*c.Padding-c.Dilation*(k-1)-1)/c.Stride + 1
	outC := c.OutChannels
	icPerGroup := inC / c.Groups
	ocPerGroup := outC / c.Groups

	out := make([]float64, n*outC*lOut)

	for ni := 0; ni < n; ni++ {
		for g := 0; g < c.Groups; g++ {
			ocStart := g * ocPerGroup
			icStart := g * icPerGroup
			for oc := ocStart; oc < ocStart+ocPerGroup; oc++ {
				// weight row for this output channel: [icPerGroup * k]
				wRow := c.Weight[oc*icPerGroup*k : (oc+1)*icPerGroup*k]
				bias := c.Bias[oc]
				for lo := 0; lo < lOut; lo++ {
					sum := bias
					// inline dot over (ic, kk)
					// Build input slice on the fly using direct loop (SIMD
					// dot product requires contiguous slice; we use the scalar
					// path here to avoid an allocation per output element).
					wIdx := 0
					for ic := 0; ic < icPerGroup; ic++ {
						absIC := icStart + ic
						for kk := 0; kk < k; kk++ {
							li := lo*c.Stride + kk*c.Dilation - c.Padding
							if li >= 0 && li < l {
								sum += x[ni*inC*l+absIC*l+li] * wRow[wIdx]
							}
							wIdx++
						}
					}
					out[ni*outC*lOut+oc*lOut+lo] = sum
				}
			}
		}
	}
	return out
}
