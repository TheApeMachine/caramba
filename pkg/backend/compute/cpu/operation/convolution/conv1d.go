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

	return conv1dForward(
		data[0], n, inC, l,
		c.Weight, c.Bias,
		c.OutChannels, c.KernelSize, c.Stride, c.Padding, c.Dilation, c.Groups,
	)
}
