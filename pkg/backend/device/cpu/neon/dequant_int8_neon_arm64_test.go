//go:build arm64

package neon

import (
	"fmt"
	"math/rand"
	"testing"
)

/*
INT8 dequant parity. NEON path: int8 → int16 (SXTL) → subtract zp →
int32 (SXTL) → f32 (SCVTF) → multiply by scale. Scalar reference:
float32(int(q) - int(zp)) * scale. Both paths use exact integer
arithmetic followed by a single f32 multiply. The conversion int32 →
f32 via SCVTF and Go's float32 cast both round-to-nearest-even.
Multiplication by scale also rounds once. Bit-exact parity is the
contract.
*/
func TestDequantInt8NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0x401d + int64(n)))
			src := make([]int8, n)
			for index := range src {
				src[index] = int8(rng.Intn(256) - 128)
			}

			scale := float32(0.0875)
			zeroPoint := int8(-13)

			scalar := make([]float32, n)
			for index, value := range src {
				scalar[index] = float32(int32(value)-int32(zeroPoint)) * scale
			}

			neon := make([]float32, n)
			DequantInt8Native(neon, src, scale, zeroPoint)

			for index := range scalar {
				if scalar[index] != neon[index] {
					t.Fatalf("N=%d lane %d scalar=%g neon=%g (q=%d zp=%d scale=%g)",
						n, index, scalar[index], neon[index],
						src[index], zeroPoint, scale,
					)
				}
			}
		})
	}
}

func BenchmarkDequantInt8NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := make([]int8, n)
			rng := rand.New(rand.NewSource(1))
			for index := range src {
				src[index] = int8(rng.Intn(256) - 128)
			}

			dst := make([]float32, n)
			scale := float32(0.0875)
			zeroPoint := int8(-13)

			b.SetBytes(int64(n * 5))
			b.ResetTimer()

			for b.Loop() {
				DequantInt8Native(dst, src, scale, zeroPoint)
			}
		})
	}
}

func BenchmarkDequantInt8Scalar(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := make([]int8, n)
			rng := rand.New(rand.NewSource(1))
			for index := range src {
				src[index] = int8(rng.Intn(256) - 128)
			}

			dst := make([]float32, n)
			scale := float32(0.0875)
			zeroPoint := int8(-13)

			b.SetBytes(int64(n * 5))
			b.ResetTimer()

			for b.Loop() {
				for index, value := range src {
					dst[index] = float32(int32(value)-int32(zeroPoint)) * scale
				}
			}
		})
	}
}
