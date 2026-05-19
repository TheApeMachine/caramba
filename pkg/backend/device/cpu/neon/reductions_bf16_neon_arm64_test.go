//go:build arm64

package neon

import (
	"fmt"
	"math"
	"testing"

	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
BF16 sum-reduction parity. Both paths widen bf16→f32 and accumulate in
f32; only the reduction order differs. With max input magnitude ~10
and n up to 8192, the worst-case relative error stays under
n * 2^-23 (f32 eps), and we cast to bf16 only at the very end where
the 7-bit mantissa of bf16 dominates the error budget. We allow 2 ULPs
of bf16 (i.e. comparing the underlying uint16 representations).
*/
func TestSumBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomBF16Slice(n, 0xD123+int64(n))

			var sumF32 float32
			for index := range src {
				sumF32 += (&src[index]).Float32()
			}

			scalar := dtype.NewBfloat16FromFloat32(sumF32)
			neon := SumBFloat16Native(src)

			scalarBits := uint16(scalar)
			neonBits := uint16(neon)
			diff := int(scalarBits) - int(neonBits)
			if diff < 0 {
				diff = -diff
			}

			if diff > 2 {
				t.Fatalf("N=%d scalar=0x%04x (%g) neon=0x%04x (%g) ulp_bf16=%d",
					n,
					scalarBits, scalar.Float32(),
					neonBits, neon.Float32(),
					diff,
				)
			}
		})
	}
}

func BenchmarkSumBFloat16NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomBF16Slice(n, 1)

			b.SetBytes(int64(n * 2))
			b.ResetTimer()

			for b.Loop() {
				_ = SumBFloat16Native(src)
			}
		})
	}
}

func BenchmarkSumBFloat16Scalar(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomBF16Slice(n, 1)

			b.SetBytes(int64(n * 2))
			b.ResetTimer()

			for b.Loop() {
				var sum float32

				for index := range src {
					sum += (&src[index]).Float32()
				}

				_ = math.Float32bits(sum)
			}
		})
	}
}
