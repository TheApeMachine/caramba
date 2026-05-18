//go:build arm64

package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestSigmoidFloat32NEONAsm(t *testing.T) {
	for _, n := range []int{1, 7, 64, 1024} {
		src := make([]float32, n)
		rng := rand.New(rand.NewSource(int64(n)))
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 3)
		}

		got := make([]float32, n)
		sigmoidFloat32NEONAsm(&got[0], &src[0], n)

		for i, x := range src {
			want := float32(1 / (1 + math.Exp(float64(-x))))
			diff := math.Abs(float64(got[i] - want))

			if diff > 1e-5 {
				t.Fatalf("sigmoid N=%d lane %d x=%g want=%g got=%g diff=%g",
					n, i, x, want, got[i], diff)
			}
		}
	}
}

func TestSiluFloat32NEONAsm(t *testing.T) {
	for _, n := range []int{1, 7, 64, 1024} {
		src := make([]float32, n)
		rng := rand.New(rand.NewSource(int64(n) + 0x1000))
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 3)
		}

		got := make([]float32, n)
		siluFloat32NEONAsm(&got[0], &src[0], n)

		for i, x := range src {
			want := x / float32(1+math.Exp(float64(-x)))
			diff := math.Abs(float64(got[i] - want))

			if diff > 1e-4 {
				t.Fatalf("silu N=%d lane %d x=%g want=%g got=%g diff=%g",
					n, i, x, want, got[i], diff)
			}
		}
	}
}

func TestTanhFloat32NEONAsm(t *testing.T) {
	for _, n := range []int{1, 7, 64, 1024} {
		src := make([]float32, n)
		rng := rand.New(rand.NewSource(int64(n) + 0x2000))
		for i := range src {
			// Keep |x| modest so 2x is in the polynomial's good range.
			src[i] = float32(rng.NormFloat64() * 1.5)
		}

		got := make([]float32, n)
		tanhFloat32NEONAsm(&got[0], &src[0], n)

		for i, x := range src {
			want := float32(math.Tanh(float64(x)))
			diff := math.Abs(float64(got[i] - want))

			if diff > 1e-4 {
				t.Fatalf("tanh N=%d lane %d x=%g want=%g got=%g diff=%g",
					n, i, x, want, got[i], diff)
			}
		}
	}
}

func BenchmarkSigmoidFloat32NEONAsm(b *testing.B) {
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
				sigmoidFloat32NEONAsm(&dst[0], &src[0], n)
			}
		})
	}
}
