package projection

import (
	"math"
	"math/rand"
)

/*
FusedQKV computes Q, K, V projections in a single matmul.

Weight layout: [(DQ+DK+DV) * DIn] row-major.
Forward:
  - shape = [M, DIn] where M = batch*seq
  - data[0] = x
  - output = x @ Weight^T = [M * (DQ+DK+DV)]  (Q, K, V concatenated)
*/
type FusedQKV struct {
	Weight []float64
	Bias   []float64
	DIn    int
	DQ     int
	DK     int
	DV     int
}

// NewFusedQKV creates a FusedQKV layer with Kaiming uniform weight init.
func NewFusedQKV(dIn, dQ, dK, dV int) *FusedQKV {
	outDim := dQ + dK + dV
	weight := make([]float64, outDim*dIn)
	bound := math.Sqrt(2.0 / float64(dIn))
	for i := range weight {
		weight[i] = (rand.Float64()*2 - 1) * bound
	}
	biasBound := 1.0 / math.Sqrt(float64(dIn))
	bias := make([]float64, outDim)
	for i := range bias {
		bias[i] = (rand.Float64()*2 - 1) * biasBound
	}
	return &FusedQKV{
		Weight: weight,
		Bias:   bias,
		DIn:    dIn,
		DQ:     dQ,
		DK:     dK,
		DV:     dV,
	}
}

// Forward computes output = x @ Weight^T [+ bias].
// Returns flat [M*(DQ+DK+DV)]; caller splits by DQ, DK, DV.
func (f *FusedQKV) Forward(shape []int, data ...[]float64) []float64 {
	M := shape[0]
	K := f.DIn
	N := f.DQ + f.DK + f.DV
	out := make([]float64, M*N)
	wT := transposeF64(f.Weight, N, K)
	applyMatmul(out, data[0], wT, M, K, N)
	if f.Bias != nil {
		addBias(out, f.Bias, M, N)
	}
	return out
}
