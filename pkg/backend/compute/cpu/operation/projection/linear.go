package projection

import (
	"math"
	"math/rand"
)

/*
Linear applies a learnable affine transformation: output = x @ Weight^T + bias.

Shape fields:
  - Weight: [OutFeatures * InFeatures] row-major (each row is one output neuron's weights)
  - Bias:   [OutFeatures] or nil

Forward:
  - shape = [batch*seq, InFeatures]
  - data[0] = x  (flattened row-major)
  - output  = [batch*seq * OutFeatures]
*/
type Linear struct {
	Weight     []float64
	Bias       []float64
	InFeatures int
	OutFeatures int
}

// NewLinear creates a Linear layer with Kaiming uniform weight initialisation.
// Bias is initialised uniformly in [-1/sqrt(InFeatures), 1/sqrt(InFeatures)].
func NewLinear(inFeatures, outFeatures int) *Linear {
	weight := make([]float64, outFeatures*inFeatures)
	bound := math.Sqrt(2.0 / float64(inFeatures))
	for i := range weight {
		weight[i] = (rand.Float64()*2 - 1) * bound
	}
	biasBound := 1.0 / math.Sqrt(float64(inFeatures))
	bias := make([]float64, outFeatures)
	for i := range bias {
		bias[i] = (rand.Float64()*2 - 1) * biasBound
	}
	return &Linear{
		Weight:      weight,
		Bias:        bias,
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
	}
}

// Forward computes output = x @ Weight^T + bias.
// shape = [M, InFeatures] where M = batch*seq.
// data[0] = flattened input x.
// Returns [M * OutFeatures].
func (l *Linear) Forward(shape []int, data ...[]float64) []float64 {
	M := shape[0]
	K := l.InFeatures
	N := l.OutFeatures
	out := make([]float64, M*N)
	// Weight is [N, K]; Weight^T is [K, N] — use Weight as B with transposed access.
	// applyMatmul computes: out[M,N] = x[M,K] @ Weight^T[K,N]
	// We treat Weight[N,K] as B and compute A @ B^T by calling our matmul
	// with a transposed copy of Weight.
	wT := transposeF64(l.Weight, N, K)
	applyMatmul(out, data[0], wT, M, K, N)
	if l.Bias != nil {
		addBias(out, l.Bias, M, N)
	}
	return out
}

// transposeF64 returns the transpose of a [rows*cols] row-major matrix as [cols*rows].
func transposeF64(src []float64, rows, cols int) []float64 {
	dst := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			dst[c*rows+r] = src[r*cols+c]
		}
	}
	return dst
}

// addBias adds bias vector [N] to each of the M rows of out [M*N].
func addBias(out, bias []float64, M, N int) {
	for i := 0; i < M; i++ {
		row := out[i*N : i*N+N]
		for j, b := range bias {
			row[j] += b
		}
	}
}
