//go:build arm64

package kernels

import (
	"math"
	"math/rand"
	"testing"
)

func TestLogFloat32NEONAsm(t *testing.T) {
	rng := rand.New(rand.NewSource(0xAAA))

	for _, n := range []int{4, 64, 1024} {
		src := make([]float32, n)
		for i := range src {
			// Positive values across a broad range.
			src[i] = float32(math.Exp(rng.NormFloat64()*2)) // ~[0.005, 200]
		}

		got := make([]float32, n)
		logFloat32NEONAsm(&got[0], &src[0], n)

		for i, x := range src {
			want := float32(math.Log(float64(x)))
			diff := math.Abs(float64(got[i] - want))

			// Polynomial-based log on f32 → expect ~5 ULP absolute.
			if diff > 1e-5 {
				t.Fatalf("N=%d lane %d x=%g want=%g got=%g diff=%g",
					n, i, x, want, got[i], diff)
			}
		}
	}
}
