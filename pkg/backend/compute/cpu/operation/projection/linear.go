package projection

import (
	"math"
	"math/rand"
)

/*
Linear applies a learnable affine transformation: output = x @ WeightT + bias.

WeightT is stored pre-transposed [InFeatures × OutFeatures] so Forward
does a single matmul with no per-call allocation.
*/
type Linear struct {
	WeightT     []float64 // [K × N] pre-transposed, K=InFeatures N=OutFeatures
	Bias        []float64
	InFeatures  int
	OutFeatures int
}

/*
NewLinear creates a Linear layer with Kaiming uniform weight initialisation.
Bias is initialised uniformly in [-1/sqrt(InFeatures), 1/sqrt(InFeatures)].
*/
func NewLinear(inFeatures, outFeatures int) *Linear {
	bound := math.Sqrt(2.0 / float64(inFeatures))
	weight := make([]float64, outFeatures*inFeatures)

	for i := range weight {
		weight[i] = (rand.Float64()*2 - 1) * bound
	}

	biasBound := 1.0 / math.Sqrt(float64(inFeatures))
	bias := make([]float64, outFeatures)

	for i := range bias {
		bias[i] = (rand.Float64()*2 - 1) * biasBound
	}

	return &Linear{
		WeightT:     transposeF64(weight, outFeatures, inFeatures),
		Bias:        bias,
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
	}
}

/*
Forward computes output = x @ WeightT + bias.
shape = [M, InFeatures] where M = batch*seq.
data[0] = flattened input x.
Returns [M * OutFeatures].
*/
func (linear *Linear) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 || len(data) < 1 || len(data[0]) < shape[0]*linear.InFeatures {
		panic("projection: Linear.Forward: invalid shape or data")
	}

	M := shape[0]
	K := linear.InFeatures
	N := linear.OutFeatures
	out := make([]float64, M*N)
	applyMatmul(out, data[0], linear.WeightT, M, K, N)

	if linear.Bias != nil {
		addBias(out, linear.Bias, M, N)
	}

	return out
}

func transposeF64(src []float64, rows, cols int) []float64 {
	dst := make([]float64, rows*cols)

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			dst[col*rows+row] = src[row*cols+col]
		}
	}

	return dst
}

func addBias(out, bias []float64, M, N int) {
	for i := 0; i < M; i++ {
		row := out[i*N : i*N+N]

		for j, b := range bias {
			row[j] += b
		}
	}
}
