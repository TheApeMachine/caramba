//go:build arm64

package reduction

import (
	"fmt"
	"math"
	"testing"
)

/*
Sum-reduction parity is not bitwise: associativity loss is intrinsic
to any parallel summation, so the order-of-operations differs from the
scalar reference. Both paths accumulate in f64 with a single final
narrowing to f32, so the running error is bounded by n*eps_f64*max|x|.
For n up to 8192 with random inputs in roughly the [-50, 50] range,
the largest absolute error is well under 2^-22 ≈ 2.4e-7 of the answer
magnitude, far tighter than a "looks fine" epsilon.
*/

func TestSumFloat32NEONAsmParity(t *testing.T) {
	for _, n := range parityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomFloat32Slice(n, 0x1A2B+int64(n))

			var sum float64

			for _, value := range src {
				sum += float64(value)
			}

			scalar := float32(sum)
			neon := SumFloat32NEONAsm(&src[0], n)

			tolerance := math.Max(math.Abs(float64(scalar)), 1.0) * float64(n) * 0x1p-50

			if math.Abs(float64(neon-scalar)) > tolerance {
				t.Fatalf("N=%d scalar=%g neon=%g diff=%g tol=%g",
					n, scalar, neon, neon-scalar, tolerance)
			}
		})
	}
}

func TestSumFloat32NEONAsmZeroLength(t *testing.T) {
	// The Go wrapper short-circuits len=0; the asm itself is never called
	// with n=0 from the dispatcher. Test the wrapper here.
	if got := SumFloat32Native(nil); got != 0 {
		t.Fatalf("sum of empty slice should be 0, got %g", got)
	}
}

func BenchmarkSumFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomFloat32Slice(n, 1)

			b.SetBytes(int64(n * 4))
			b.ResetTimer()

			for b.Loop() {
				_ = SumFloat32NEONAsm(&src[0], n)
			}
		})
	}
}

func BenchmarkSumFloat32Scalar(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomFloat32Slice(n, 1)

			b.SetBytes(int64(n * 4))
			b.ResetTimer()

			for b.Loop() {
				var sum float64

				for _, value := range src {
					sum += float64(value)
				}

				_ = float32(sum)
			}
		})
	}
}
