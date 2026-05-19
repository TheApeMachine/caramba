//go:build arm64

package dot

import (
	"fmt"
	"testing"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestDotFloat16NEONAsmParity(t *testing.T) {
	for _, n := range parityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			a := randomF16Slice(n, 0xE001+int64(n))
			b := randomF16Slice(n, 0xF002+int64(n))

			var sumF32 float32
			for index := range a {
				sumF32 += a[index].Float32() * b[index].Float32()
			}

			scalar := dtype.Fromfloat32(sumF32)
			neon := DotFloat16Native(a, b)

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

func BenchmarkDotFloat16NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			x := randomF16Slice(n, 1)
			y := randomF16Slice(n, 2)

			b.SetBytes(int64(n * 2 * 2))
			b.ResetTimer()

			for b.Loop() {
				_ = DotFloat16Native(x, y)
			}
		})
	}
}
