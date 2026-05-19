//go:build arm64

package dropout

import (
	"math"
	"testing"
)

func TestDropoutFloat32NEONAsm(t *testing.T) {
	// keepProb = 0.5 → threshold = 0.5 * 2^32 = 0x80000000.
	const n = 4096
	src := make([]float32, n)
	dst := make([]float32, n)
	for i := range src {
		src[i] = 1.0
	}

	seed := []uint32{1, 2, 3, 4}
	scale := float32(2.0) // 1 / 0.5
	// Threshold passes through as a raw 32-bit pattern; the asm
	// reinterprets it as uint32 for the CMHI compare. Encode the
	// uint32 0x80000000 (≈ 50% keep rate) via math.Float32frombits.
	threshold := math.Float32frombits(0x80000000)

	DropoutFloat32NEONAsm(&dst[0], &src[0], n, &seed[0], scale, threshold)

	// Count kept lanes; with keep=0.5 we expect ~n/2 nonzero.
	var kept int
	for _, v := range dst {
		if v != 0 {
			kept++
			if math.Abs(float64(v-2.0)) > 1e-6 {
				t.Fatalf("kept lane has wrong value %g, expected 2.0", v)
			}
		}
	}

	// Allow ~10% variance from the binomial expectation.
	expected := n / 2
	if kept < expected*9/10 || kept > expected*11/10 {
		t.Fatalf("kept count %d out of expected range [%d, %d]",
			kept, expected*9/10, expected*11/10)
	}

	t.Logf("dropout kept %d/%d lanes at keepProb=0.5", kept, n)
}
