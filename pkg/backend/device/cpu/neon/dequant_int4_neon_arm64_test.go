//go:build arm64

package neon

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
INT4 dequant parity. NEON unpacks packed nibbles, sign-extends, runs
through the same int8→int16→int32→f32→×scale pipeline as int8. Scalar
reference uses (int(nibble) - int(zp)) * scale on each logical index
read through Int4Vector.Get.
*/

func TestDequantInt4NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0x401e + int64(n)))
			byteCount := (n + 1) / 2
			bytes := make([]byte, byteCount)
			for index := range bytes {
				bytes[index] = byte(rng.Uint32())
			}

			pairs := tensor.NewInt4Vector(asInt4PairSlice(bytes), n)

			scale := float32(0.0625)
			zeroPoint := int8(3)

			scalar := make([]float32, n)
			for index := range scalar {
				nibble := pairs.Get(index)
				scalar[index] = float32(int(nibble)-int(zeroPoint)) * scale
			}

			neon := make([]float32, n)
			DequantInt4Native(neon, pairs, scale, zeroPoint)

			for index := range scalar {
				if scalar[index] != neon[index] {
					t.Fatalf("N=%d lane %d scalar=%g neon=%g (nibble=%d zp=%d scale=%g)",
						n, index, scalar[index], neon[index],
						pairs.Get(index), zeroPoint, scale,
					)
				}
			}
		})
	}
}

func asInt4PairSlice(bytes []byte) []dtype.Int4Pair {
	pairs := make([]dtype.Int4Pair, len(bytes))
	for index, b := range bytes {
		pairs[index] = dtype.Int4Pair(b)
	}
	return pairs
}

func BenchmarkDequantInt4NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			byteCount := (n + 1) / 2
			bytes := make([]byte, byteCount)
			rng := rand.New(rand.NewSource(1))
			for index := range bytes {
				bytes[index] = byte(rng.Uint32())
			}

			pairs := tensor.NewInt4Vector(asInt4PairSlice(bytes), n)
			dst := make([]float32, n)
			scale := float32(0.0625)
			zeroPoint := int8(3)

			b.SetBytes(int64(n / 2))
			b.ResetTimer()

			for b.Loop() {
				DequantInt4Native(dst, pairs, scale, zeroPoint)
			}
		})
	}
}
