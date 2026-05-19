//go:build arm64

package neon

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

/*
INT8 quant parity. NEON path uses FCVTAS (round-to-nearest, ties away
from zero) which matches math.Round in the scalar reference.
SQXTN/SQXTN2 perform signed saturating narrow at each step, matching
the explicit clamp in the scalar code.
*/

func TestQuantInt8NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0x55aa + int64(n)))
			src := make([]float32, n)
			for index := range src {
				// Random values across the typical int8 range scaled by scale.
				src[index] = float32(rng.NormFloat64()) * 10
			}

			scale := float32(0.125)
			zeroPoint := int8(7)

			scalar := make([]int8, n)
			for index, value := range src {
				scaled := math.Round(float64(value/scale)) + float64(zeroPoint)

				switch {
				case scaled > float64(math.MaxInt8):
					scalar[index] = math.MaxInt8
				case scaled < float64(math.MinInt8):
					scalar[index] = math.MinInt8
				default:
					scalar[index] = int8(scaled)
				}
			}

			neon := make([]int8, n)
			QuantInt8Native(neon, src, scale, zeroPoint)

			for index := range scalar {
				if scalar[index] != neon[index] {
					t.Fatalf("N=%d lane %d scalar=%d neon=%d (src=%g scale=%g zp=%d)",
						n, index, scalar[index], neon[index],
						src[index], scale, zeroPoint,
					)
				}
			}
		})
	}
}

func BenchmarkQuantInt8NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := make([]float32, n)
			rng := rand.New(rand.NewSource(1))
			for index := range src {
				src[index] = float32(rng.NormFloat64()) * 10
			}

			dst := make([]int8, n)
			scale := float32(0.125)
			zeroPoint := int8(7)

			b.SetBytes(int64(n * 5))
			b.ResetTimer()

			for b.Loop() {
				QuantInt8Native(dst, src, scale, zeroPoint)
			}
		})
	}
}
