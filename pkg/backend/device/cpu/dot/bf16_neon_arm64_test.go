//go:build arm64

package dot

import (
	"fmt"
	"testing"

	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
BF16 dot product parity. Both paths multiply pairs in f32 and
accumulate in f32 with a final cast to bf16. The reduction order
differs between NEON's parallel accumulators and the sequential scalar
loop, so we allow up to 2 ULPs at the bf16 representation level
(which is far tighter than any tolerance an FMA-vs-not-FMA divergence
could produce).
*/
func TestDotBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range parityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			a := randomBF16Slice(n, 0xE234+int64(n))
			b := randomBF16Slice(n, 0xF345+int64(n))

			var sumF32 float32
			for index := range a {
				sumF32 += (&a[index]).Float32() * (&b[index]).Float32()
			}

			scalar := dtype.NewBfloat16FromFloat32(sumF32)
			neon := DotBFloat16Native(a, b)

			diff := int(uint16(scalar)) - int(uint16(neon))
			if diff < 0 {
				diff = -diff
			}

			if diff > 2 {
				t.Fatalf("N=%d scalar=0x%04x (%g) neon=0x%04x (%g) ulp_bf16=%d",
					n,
					uint16(scalar), scalar.Float32(),
					uint16(neon), neon.Float32(),
					diff,
				)
			}
		})
	}
}

func BenchmarkDotBFloat16NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			x := randomBF16Slice(n, 1)
			y := randomBF16Slice(n, 2)

			b.SetBytes(int64(n * 2 * 2))
			b.ResetTimer()

			for b.Loop() {
				_ = DotBFloat16Native(x, y)
			}
		})
	}
}
