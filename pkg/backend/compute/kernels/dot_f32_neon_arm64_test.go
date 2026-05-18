//go:build arm64

package kernels

import (
	"fmt"
	"math"
	"testing"
)

/*
Dot-product parity uses the same f64-throughout precision model as
sum reduction: each product is widened-multiplied in f64, accumulated
in f64, then narrowed to f32 once at the end. Both paths therefore
have an error bound of O(n*eps_f64*max|a_i|*max|b_i|), with only the
reduction order differing. Tolerance is set to n * 2^-50 times the
scalar magnitude — extremely tight for the test sizes.
*/

func TestDotFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			a := randomFloat32Slice(n, 0xDEAD+int64(n))
			b := randomFloat32Slice(n, 0xBEEF+int64(n))

			var sum float64

			for index := range a {
				sum += float64(a[index]) * float64(b[index])
			}

			scalar := float32(sum)
			neon := dotFloat32NEONAsm(&a[0], &b[0], n)

			tolerance := math.Max(math.Abs(float64(scalar)), 1.0) * float64(n) * 0x1p-50

			if math.Abs(float64(neon-scalar)) > tolerance {
				t.Fatalf("N=%d scalar=%g neon=%g diff=%g tol=%g",
					n, scalar, neon, neon-scalar, tolerance)
			}
		})
	}
}

func BenchmarkDotFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			a := randomFloat32Slice(n, 1)
			c := randomFloat32Slice(n, 2)

			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()

			for b.Loop() {
				_ = dotFloat32NEONAsm(&a[0], &c[0], n)
			}
		})
	}
}

func BenchmarkDotFloat32Scalar(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			a := randomFloat32Slice(n, 1)
			c := randomFloat32Slice(n, 2)

			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()

			for b.Loop() {
				var sum float64

				for index := range a {
					sum += float64(a[index]) * float64(c[index])
				}

				_ = float32(sum)
			}
		})
	}
}
