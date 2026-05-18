//go:build arm64

package kernels

import (
	"math"
	"math/rand"
	"testing"
)

func TestConv2DStride1RowNEONAsm(t *testing.T) {
	// Small parametrized test: B=1, inC=3, inH=8, inW=8, outC=1, kH=3, kW=3.
	const (
		inC   = 3
		inH   = 8
		inW   = 8
		kH    = 3
		kW    = 3
		outH  = inH - kH + 1
		outW  = inW - kW + 1
	)

	rng := rand.New(rand.NewSource(0xC033))
	input := make([]float32, inC*inH*inW)
	weight := make([]float32, inC*kH*kW)
	for i := range input {
		input[i] = float32(rng.NormFloat64())
	}
	for i := range weight {
		weight[i] = float32(rng.NormFloat64())
	}
	bias := float32(rng.NormFloat64())

	// Scalar reference for the single (b=0, oc=0) row at oh=0.
	scalar := make([]float32, outW)
	for ow := 0; ow < outW; ow++ {
		var sum float32 = bias
		for ic := 0; ic < inC; ic++ {
			for kh := 0; kh < kH; kh++ {
				for kw := 0; kw < kW; kw++ {
					ih := 0 + kh
					iw := ow + kw
					sum += input[ic*inH*inW+ih*inW+iw] * weight[ic*kH*kW+kh*kW+kw]
				}
			}
		}
		scalar[ow] = sum
	}

	// NEON path requires outCols divisible by 4. For this test outW=6,
	// so call NEON for first 4 columns and check parity, then scalar
	// for the remainder externally.
	got := make([]float32, 4)
	conv2dStride1RowNEONAsm(
		&got[0],
		&input[0],
		&weight[0],
		bias,
		4,
		inC, kH, kW,
		inW, inH*inW,
		kW, kH*kW,
		0, 0,
	)

	for i := 0; i < 4; i++ {
		diff := math.Abs(float64(got[i] - scalar[i]))
		if diff > 1e-4 {
			t.Fatalf("col %d scalar=%g neon=%g diff=%g", i, scalar[i], got[i], diff)
		}
	}
}
