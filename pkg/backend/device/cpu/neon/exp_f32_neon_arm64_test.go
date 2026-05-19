//go:build arm64

package neon

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func BenchmarkExpFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{1024, 8192} {
		n := n
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := make([]float32, n)
			dst := make([]float32, n)
			rng := rand.New(rand.NewSource(1))
			for i := range src {
				src[i] = float32(rng.NormFloat64() * 3)
			}
			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()
			for b.Loop() {
				ExpFloat32NEONAsm(&dst[0], &src[0], n)
			}
		})
	}
}

func BenchmarkExpFloat32Scalar(b *testing.B) {
	for _, n := range []int{1024, 8192} {
		n := n
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := make([]float32, n)
			dst := make([]float32, n)
			rng := rand.New(rand.NewSource(1))
			for i := range src {
				src[i] = float32(rng.NormFloat64() * 3)
			}
			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()
			for b.Loop() {
				for i, x := range src {
					dst[i] = float32(math.Exp(float64(x)))
				}
			}
		})
	}
}

func TestExpFloat32NEONAsm(t *testing.T) {
	rng := rand.New(rand.NewSource(0xEEE))

	// Stay in the range where exp(x) is comfortably finite in f32.
	cases := []int{1, 7, 64, 1024}
	for _, n := range cases {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 3)
		}

		got := make([]float32, n)
		ExpFloat32NEONAsm(&got[0], &src[0], n)

		for i, x := range src {
			want := float32(math.Exp(float64(x)))
			diff := math.Abs(float64(got[i]-want)) / math.Abs(float64(want))

			// 5th-order polynomial after range reduction → ~3 ULP relative.
			if diff > 1e-5 {
				t.Fatalf("N=%d lane %d x=%g want=%g got=%g rel=%g",
					n, i, x, want, got[i], diff)
			}
		}
	}
}
