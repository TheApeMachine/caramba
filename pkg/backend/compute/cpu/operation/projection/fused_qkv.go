package projection

import (
	"math"
	"math/rand"
	"time"
)

/*
FusedQKV computes Q, K, V projections in a single matmul.

WeightT is stored pre-transposed [DIn × (DQ+DK+DV)] to avoid per-call
transpose allocation. Forward computes x @ WeightT directly.
*/
type FusedQKV struct {
	WeightT []float64 // [DIn × (DQ+DK+DV)] pre-transposed
	Bias    []float64
	DIn     int
	DQ      int
	DK      int
	DV      int
}

/*
NewFusedQKV creates a FusedQKV layer with Kaiming uniform weight init.
*/
func NewFusedQKV(dIn, dQ, dK, dV int) *FusedQKV {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	outDim := dQ + dK + dV
	bound := math.Sqrt(2.0 / float64(dIn))
	weight := make([]float64, outDim*dIn)

	for i := range weight {
		weight[i] = (rng.Float64()*2 - 1) * bound
	}

	biasBound := 1.0 / math.Sqrt(float64(dIn))
	bias := make([]float64, outDim)

	for i := range bias {
		bias[i] = (rng.Float64()*2 - 1) * biasBound
	}

	return &FusedQKV{
		WeightT: transposeF64(weight, outDim, dIn),
		Bias:    bias,
		DIn:     dIn,
		DQ:      dQ,
		DK:      dK,
		DV:      dV,
	}
}

/*
Forward computes output = x @ WeightT [+ bias].
Returns flat [M*(DQ+DK+DV)]; caller splits by DQ, DK, DV.
*/
func (fqkv *FusedQKV) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 || len(data) < 1 || len(data[0]) < shape[0]*fqkv.DIn {
		panic("projection: FusedQKV.Forward: invalid shape or data")
	}

	M := shape[0]
	K := fqkv.DIn
	N := fqkv.DQ + fqkv.DK + fqkv.DV
	out := make([]float64, M*N)
	applyMatmul(out, data[0], fqkv.WeightT, M, K, N)

	if fqkv.Bias != nil {
		addBias(out, fqkv.Bias, M, N)
	}

	return out
}
