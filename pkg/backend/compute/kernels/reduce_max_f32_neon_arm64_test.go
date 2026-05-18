//go:build arm64

package kernels

import (
	"fmt"
	"math"
	"testing"
)

/*
reduce_max parity. NEON path uses FMAX (maxnum semantics — returns
the non-NaN operand if exactly one is NaN). Scalar reference uses
`if v > running { running = v }` which is NaN-non-propagating. The two
agree exactly on finite inputs; we restrict the random generator to
finite values for parity.
*/

func TestReduceMaxFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomFloat32Slice(n, 0xA22A+int64(n))

			scalar := src[0]
			for _, value := range src[1:] {
				if value > scalar {
					scalar = value
				}
			}

			neon := reduceMaxFloat32Native(src)

			if math.Float32bits(scalar) != math.Float32bits(neon) {
				t.Fatalf("N=%d scalar=%g (0x%08x) neon=%g (0x%08x)",
					n, scalar, math.Float32bits(scalar),
					neon, math.Float32bits(neon),
				)
			}
		})
	}
}

func BenchmarkReduceMaxFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomFloat32Slice(n, 1)
			b.SetBytes(int64(n * 4))
			b.ResetTimer()

			for b.Loop() {
				_ = reduceMaxFloat32Native(src)
			}
		})
	}
}
