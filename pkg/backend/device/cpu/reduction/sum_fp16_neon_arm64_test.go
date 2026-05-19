//go:build arm64

package reduction

import (
	"fmt"
	"testing"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestSumFloat16NEONAsmParity(t *testing.T) {
	for _, n := range parityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomF16Slice(n, 0xF111+int64(n))

			var sumF32 float32
			for index := range src {
				sumF32 += src[index].Float32()
			}

			scalar := dtype.Fromfloat32(sumF32)
			neon := SumFloat16Native(src)

			diff := int(uint16(scalar)) - int(uint16(neon))
			if diff < 0 {
				diff = -diff
			}

			if diff > 2 {
				t.Fatalf("N=%d scalar=0x%04x (%g) neon=0x%04x (%g) ulp_f16=%d",
					n,
					uint16(scalar), scalar.Float32(),
					uint16(neon), neon.Float32(),
					diff,
				)
			}
		})
	}
}

func BenchmarkSumFloat16NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomF16Slice(n, 1)

			b.SetBytes(int64(n * 2))
			b.ResetTimer()

			for b.Loop() {
				_ = SumFloat16Native(src)
			}
		})
	}
}
