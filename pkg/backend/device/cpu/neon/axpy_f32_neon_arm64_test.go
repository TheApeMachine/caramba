//go:build arm64

package neon

import (
	"fmt"
	"math"
	"testing"
)

/*
AXPY parity. The NEON kernel uses a fused multiply-add (FMA): each
output lane is computed as round_to_f32(alpha*x + y) with a single
rounding step. A naive Go expression `y + alpha*x` performs two
roundings (multiply then add) and can differ from the FMA result by
up to 1 ULP at the output magnitude.

The scalar reference therefore uses math.FMA on widened f64 operands,
which gives a single-rounded result in f64, then casts to f32. That
casting introduces a second rounding for one of the two paths, so the
worst-case divergence remains at 1 ULP of the output. We allow 2 ULPs
to give a margin for the double-rounding edge case and keep the bound
tight.
*/

func TestAxpyFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			yInit := randomFloat32Slice(n, 0xAA00+int64(n))
			x := randomFloat32Slice(n, 0xBB00+int64(n))
			alpha := float32(0.7654321)

			expected := make([]float32, n)

			for index := range yInit {
				expected[index] = float32(math.FMA(float64(alpha), float64(x[index]), float64(yInit[index])))
			}

			actual := append([]float32(nil), yInit...)
			AxpyFloat32NEONAsm(&actual[0], &x[0], alpha, n)

			for index := range expected {
				diffULP := ulpDistanceFloat32(expected[index], actual[index])

				if diffULP > 2 {
					t.Fatalf("N=%d lane %d expected=%g (0x%08x) actual=%g (0x%08x) ulp=%d",
						n, index,
						expected[index], math.Float32bits(expected[index]),
						actual[index], math.Float32bits(actual[index]),
						diffULP,
					)
				}
			}
		})
	}
}

func ulpDistanceFloat32(a, b float32) uint32 {
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		// NaN sentinels are out of scope for this test.
		return 0
	}

	bitsA := math.Float32bits(a)
	bitsB := math.Float32bits(b)

	// Sign flip handling: map negative floats to a continuous integer
	// ordering by xoring with 0x7FFFFFFF when the sign bit is set.
	if bitsA&0x80000000 != 0 {
		bitsA = 0x80000000 - bitsA
	}

	if bitsB&0x80000000 != 0 {
		bitsB = 0x80000000 - bitsB
	}

	if bitsA > bitsB {
		return bitsA - bitsB
	}

	return bitsB - bitsA
}

func BenchmarkAxpyFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			y := randomFloat32Slice(n, 1)
			x := randomFloat32Slice(n, 2)
			alpha := float32(0.5)

			b.SetBytes(int64(n * 4 * 3))
			b.ResetTimer()

			for b.Loop() {
				AxpyFloat32NEONAsm(&y[0], &x[0], alpha, n)
			}
		})
	}
}

func BenchmarkAxpyFloat32Scalar(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			y := randomFloat32Slice(n, 1)
			x := randomFloat32Slice(n, 2)
			alpha := float32(0.5)

			b.SetBytes(int64(n * 4 * 3))
			b.ResetTimer()

			for b.Loop() {
				for index := range y {
					y[index] += alpha * x[index]
				}
			}
		})
	}
}
