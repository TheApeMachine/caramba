//go:build arm64

package neon

import (
	"math"
	"math/rand"
	"testing"
)

func TestMseSumNEONAsm(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5E))
	for _, n := range []int{4, 17, 256, 1023} {
		pred := make([]float32, n)
		targ := make([]float32, n)
		for i := range pred {
			pred[i] = float32(rng.NormFloat64())
			targ[i] = float32(rng.NormFloat64())
		}

		var want float64
		for i := range pred {
			d := float64(pred[i] - targ[i])
			want += d * d
		}

		got := MseSumNEONAsm(&pred[0], &targ[0], n)
		diff := math.Abs(float64(got) - want)
		if diff/math.Max(want, 1) > 1e-5 {
			t.Fatalf("N=%d want=%g got=%g", n, want, got)
		}
	}
}

func TestL1NormNEONAsm(t *testing.T) {
	rng := rand.New(rand.NewSource(0xAB))
	for _, n := range []int{4, 17, 256, 1023} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64())
		}

		var want float64
		for _, v := range src {
			want += math.Abs(float64(v))
		}

		got := L1NormNEONAsm(&src[0], n)
		diff := math.Abs(float64(got) - want)
		if diff/math.Max(want, 1) > 1e-5 {
			t.Fatalf("N=%d want=%g got=%g", n, want, got)
		}
	}
}
