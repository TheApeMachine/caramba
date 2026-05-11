package projection

import (
	"fmt"
	"math"
	"math/rand"
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
NewFusedQKV creates a FusedQKV layer with Kaiming uniform weight init using an implicit deterministic seed (see NewFusedQKVWithSeed).
*/
func NewFusedQKV(dIn, dQ, dK, dV int) *FusedQKV {
	return NewFusedQKVWithSeed(dIn, dQ, dK, dV, 1)
}

/*
NewFusedQKVWithSeed creates a FusedQKV with Kaiming uniform weight init driven by the given seed.
*/
func NewFusedQKVWithSeed(dIn, dQ, dK, dV int, seed int64) *FusedQKV {
	return NewFusedQKVWithRNG(dIn, dQ, dK, dV, rand.New(rand.NewSource(seed)))
}

/*
NewFusedQKVWithRNG creates a FusedQKV using the caller-owned RNG for weight and bias init.
*/
func NewFusedQKVWithRNG(dIn, dQ, dK, dV int, rng *rand.Rand) *FusedQKV {
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
	N := fqkv.DQ + fqkv.DK + fqkv.DV

	if len(shape) < 1 {
		panic("projection: FusedQKV.Forward: empty shape")
	}

	M := shape[0]

	if M <= 0 {
		panic(fmt.Sprintf("projection: FusedQKV.Forward: shape[0]=%d must be > 0", M))
	}

	if fqkv.DIn <= 0 {
		panic(fmt.Sprintf("projection: FusedQKV.Forward: DIn=%d must be > 0", fqkv.DIn))
	}

	if len(data) < 1 || data[0] == nil {
		panic("projection: FusedQKV.Forward: missing data[0]")
	}

	if M > len(data[0])/fqkv.DIn {
		panic(fmt.Sprintf(
			"projection: FusedQKV.Forward: len(data[0])=%d insufficient for shape[0]=%d and DIn=%d (need len(data[0]) >= shape[0]*DIn)",
			len(data[0]), M, fqkv.DIn,
		))
	}

	wantW := int64(fqkv.DIn) * int64(N)

	if wantW < 0 || wantW > int64(math.MaxInt) || len(fqkv.WeightT) != int(wantW) {
		panic(fmt.Sprintf(
			"projection: FusedQKV.Forward: len(WeightT)=%d want DIn*(DQ+DK+DV)=%d (DIn=%d N=%d)",
			len(fqkv.WeightT), int(wantW), fqkv.DIn, N,
		))
	}

	if fqkv.Bias != nil && len(fqkv.Bias) != N {
		panic(fmt.Sprintf(
			"projection: FusedQKV.Forward: len(Bias)=%d want DQ+DK+DV=%d",
			len(fqkv.Bias), N,
		))
	}

	if int64(M)*int64(N) < 0 || int64(M)*int64(N) > int64(math.MaxInt) {
		panic(fmt.Sprintf("projection: FusedQKV.Forward: M*N overflows int (M=%d N=%d)", M, N))
	}

	K := fqkv.DIn
	out := make([]float64, M*N)
	applyMatmul(out, data[0], fqkv.WeightT, M, K, N)

	if fqkv.Bias != nil {
		addBias(out, fqkv.Bias, M, N)
	}

	return out
}
